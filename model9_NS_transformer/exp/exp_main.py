from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping
from utils.metrics import metric
import gc
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.diffusion_models.diffusion_utils import *
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler
import os
import time
from utils.metrics import calc_quantile_CRPS_sum
from multiprocessing import Pool
import CRPS.CRPS as pscore
from layers.Decompose import moving_avg
import warnings
from layers.Decompose import series_decomp, FourierLayer
from model9_NS_transformer.denoise_models.residual_heads import (
    ResidualLocationScaleHead,
    gaussian_nll_residual_loss,
    standardize_residual,
    destandardize_residual,
)

warnings.filterwarnings('ignore')


def ccc(id, pred, true):
    res_box = np.zeros(len(true))

    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var
    )


def calculate_crps_sum_worker(args):
    pred, true = args
    p_in = np.sum(pred, axis=-1).T
    t_in = np.sum(true, axis=-1).reshape(-1)
    crps = ccc(8, p_in, t_in)
    return crps.mean()


def calculate_crps_worker(args):
    pred, true = args
    p_in = pred.transpose(1, 0, 2)
    t_in = true
    all_res = []
    for i in range(pred.shape[-1]):
        crps = ccc(8, p_in[:, :, i], t_in[:, i])
        all_res.append(crps)
    all_res = np.array(all_res)
    if isinstance(all_res, np.ndarray):
        return np.mean(all_res, axis=0).mean()
    else:
        return all_res


class HRD3UCheckpointWrapper(nn.Module):
    """
    Save/load diffusion model and residual location-scale head together
    while staying compatible with the existing EarlyStopping utility.
    """
    def __init__(self, diffusion_model, residual_head=None):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.residual_head = residual_head

    def state_dict(self, *args, **kwargs):
        state = {"model": self.diffusion_model.state_dict(*args, **kwargs)}
        if self.residual_head is not None:
            state["residual_head"] = self.residual_head.state_dict(*args, **kwargs)
        return state

#
class ResidualDiagnostics:
    def __init__(self):
        self.residual_list = []
        self.mu_list = []
        self.sigma_list = []
        self.z_list = []
        self.y_list = []
        self.y_base_list = []

    @torch.no_grad()
    def update(self, y, y_base, mu, sigma):
        """
        y:      [B, pred_len, C]
        y_base: [B, pred_len, C]
        mu:     [B, pred_len, C]
        sigma:  [B, pred_len, C]
        """
        residual = y - y_base
        z = (residual - mu) / (sigma + 1e-8)

        self.residual_list.append(residual.detach().cpu().reshape(-1))
        self.mu_list.append(mu.detach().cpu().reshape(-1))
        self.sigma_list.append(sigma.detach().cpu().reshape(-1))
        self.z_list.append(z.detach().cpu().reshape(-1))
        self.y_list.append(y.detach().cpu().reshape(-1))
        self.y_base_list.append(y_base.detach().cpu().reshape(-1))

    def summarize(self, prefix="Residual Diagnostics"):
        if len(self.residual_list) == 0:
            print("[ResidualDiagnostics] No data collected.")
            return

        residual = torch.cat(self.residual_list)
        mu = torch.cat(self.mu_list)
        sigma = torch.cat(self.sigma_list)
        z = torch.cat(self.z_list)
        y = torch.cat(self.y_list)
        y_base = torch.cat(self.y_base_list)

        y_base_plus_mu = y_base + mu

        mse_base = torch.mean((y - y_base) ** 2).item()
        mae_base = torch.mean(torch.abs(y - y_base)).item()

        mse_mu = torch.mean((y - y_base_plus_mu) ** 2).item()
        mae_mu = torch.mean(torch.abs(y - y_base_plus_mu)).item()

        abs_residual = torch.abs(residual)
        corr = torch.mean(
            (abs_residual - abs_residual.mean()) *
            (sigma - sigma.mean())
        ) / (
            torch.std(abs_residual) * torch.std(sigma) + 1e-8
        )

        print("=" * 80)
        print(prefix)
        print("-" * 80)
        print(f"residual mean: {residual.mean().item():.6f}")
        print(f"residual std : {residual.std().item():.6f}")
        print(f"mu_r mean    : {mu.mean().item():.6f}")
        print(f"mu_r std     : {mu.std().item():.6f}")
        print(f"sigma_r mean : {sigma.mean().item():.6f}")
        print(f"sigma_r std  : {sigma.std().item():.6f}")
        print(f"sigma_r min  : {sigma.min().item():.6f}")
        print(f"sigma_r max  : {sigma.max().item():.6f}")
        print(f"z mean       : {z.mean().item():.6f}")
        print(f"z std        : {z.std().item():.6f}")
        print(f"corr(|residual|, sigma_r): {corr.item():.6f}")
        print("-" * 80)
        print(f"MSE(y_base, y)        : {mse_base:.6f}")
        print(f"MAE(y_base, y)        : {mae_base:.6f}")
        print(f"MSE(y_base + mu_r, y) : {mse_mu:.6f}")
        print(f"MAE(y_base + mu_r, y) : {mae_mu:.6f}")
        print(f"MSE improvement       : {(mse_base - mse_mu) / (mse_base + 1e-8) * 100:.2f}%")
        print(f"MAE improvement       : {(mae_base - mae_mu) / (mae_base + 1e-8) * 100:.2f}%")
        print("=" * 80)
#

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.moving_avg = moving_avg(7, stride=1)
        self.decomp = series_decomp(kernel_size=15)
        self.seasonal = FourierLayer(d_model=128, factor=1)

    def _build_model(self):
        model = diffuMTS.Model(self.args).float()

        cond_pred_model = self.cond_model_dict[self.args.model].Model(self.args).float()
        condition_path = os.path.join(self.args.pretrain_checkpoints, self.args.model)
        if self.args.decomposition:
            best_condition_model_path = (
                condition_path + '/' + str('decomposition') + '/' +
                self.args.data_name + '/' + str(self.args.pred_len) + '/' + 'checkpoint.pth'
            )
        else:
            best_condition_model_path = (
                condition_path + '/' + str('all') + '/' +
                self.args.data_name + '/' + str(self.args.pred_len) + '/' + 'checkpoint.pth'
            )
        print(best_condition_model_path)

        if self.args.from_scrach == False:
            cond_pred_model.load_state_dict(torch.load(best_condition_model_path, map_location=self.device))
            if self.args.cond_pred_model_requires_grad:
                for param in cond_pred_model.parameters():
                    param.requires_grad = True
            else:
                for param in cond_pred_model.parameters():
                    param.requires_grad = False

        self.residual_head = None
        if getattr(self.args, "use_hrd3u", False):
            self.residual_head = ResidualLocationScaleHead(
                self.args,
                hidden_size=self.args.residual_head_hidden,
                dropout=self.args.residual_head_dropout,
                sigma_min=self.args.residual_sigma_min,
                sigma_max=self.args.residual_sigma_max,
                mu_clip=self.args.residual_mu_clip,
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)
            if self.residual_head is not None:
                self.residual_head = nn.DataParallel(self.residual_head, device_ids=self.args.device_ids)

        return model, cond_pred_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            params = [{'params': self.model.parameters()}]
            if self.residual_head is not None and not getattr(self.args, "freeze_residual_head_after_pretrain", False):
                params.append({'params': self.residual_head.parameters(), 'lr': self.args.residual_head_lr})
            params.append({'params': self.cond_pred_model.parameters()})
            model_optim = optim.Adam(params, lr=self.args.learning_rate)
        elif mode == 'ResidualHead':
            if self.residual_head is None:
                return None
            model_optim = optim.Adam(
                self.residual_head.parameters(),
                lr=self.args.residual_head_lr,
                weight_decay=self.args.residual_head_weight_decay,
            )
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_condition_outputs(self, batch_x, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        y_T_mean, _, enc_out = self.cond_pred_model(batch_x, None, dec_inp, None)
        if self.args.use_pretraining_condition:
            enc_out = y_T_mean
        return y_T_mean, enc_out

    def _make_diffusion_target(self, batch_y, y_T_mean, enc_out):
        batch_y = batch_y[:, -self.args.pred_len:, :]

        if self.args.bias:
            y_0 = batch_y - y_T_mean

            if self.args.bias_y_0:
                res, trend = self.decomp(batch_y)
                seasonal = self.seasonal(res)
                y_0 = res - seasonal
        else:
            y_0 = batch_y

        mu_r, sigma_r = None, None

        if getattr(self.args, "use_hrd3u", False):

            # HR-D3U 只处理 normal residual:
            # r = y - y_base
            if self.args.bias and not self.args.bias_y_0:

                mu_r, sigma_r = self.residual_head(enc_out)

                mu_r = mu_r.to(y_0.device)
                sigma_r = sigma_r.to(y_0.device)

                # standardized residual
                y_0 = standardize_residual(y_0, mu_r, sigma_r)

        return batch_y, y_0, mu_r, sigma_r

    def _pretrain_residual_head(self, train_loader, vali_loader):
        if self.residual_head is None:
            return

        print(">>>>>>>pretraining residual location-scale head>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.model.eval()
        self.cond_pred_model.eval()
        self.residual_head.train()

        optimizer = self._select_optimizer(mode='ResidualHead')
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.args.residual_head_epochs):
            train_losses = []
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                with torch.no_grad():
                    y_T_mean, enc_out = self._get_condition_outputs(batch_x, batch_y)
                    batch_y_target = batch_y[:, -self.args.pred_len:, :]
                    residual = batch_y_target - y_T_mean

                mu_r, sigma_r = self.residual_head(enc_out)
                mu_r = mu_r.to(residual.device)
                sigma_r = sigma_r.to(residual.device)
                loss = gaussian_nll_residual_loss(residual, mu_r, sigma_r)

                # Optional weak regularization: keep standardized residual close to unit scale.
                if self.args.residual_z_reg > 0:
                    z = (residual - mu_r) / sigma_r
                    z_reg = (z.mean().square() + (z.std(unbiased=False) - 1.0).square())
                    loss = loss + self.args.residual_z_reg * z_reg

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss = self._vali_residual_head(vali_loader)
            print(
                "ResidualHead Epoch: {0}, cost time: {1:.2f}s | Train NLL: {2:.7f} Val NLL: {3:.7f}".format(
                    epoch + 1, time.time() - epoch_time, np.average(train_losses), val_loss
                )
            )

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.residual_head.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.args.residual_head_patience:
                    print("Residual head early stopping")
                    break

        if best_state is not None:
            self.residual_head.load_state_dict(best_state)

        if self.args.freeze_residual_head_after_pretrain:
            self.residual_head.eval()
            for p in self.residual_head.parameters():
                p.requires_grad = False

    def _vali_residual_head(self, vali_loader):
        self.cond_pred_model.eval()
        self.residual_head.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                y_T_mean, enc_out = self._get_condition_outputs(batch_x, batch_y)
                batch_y_target = batch_y[:, -self.args.pred_len:, :]
                residual = batch_y_target - y_T_mean

                mu_r, sigma_r = self.residual_head(enc_out)
                mu_r = mu_r.to(residual.device)
                sigma_r = sigma_r.to(residual.device)
                loss = gaussian_nll_residual_loss(residual, mu_r, sigma_r)
                total_loss.append(loss.detach().cpu().item())

        self.residual_head.train()
        return np.average(total_loss)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.cond_pred_model.eval()
        if self.residual_head is not None:
            self.residual_head.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

                y_T_mean, enc_out = self._get_condition_outputs(batch_x, batch_y)
                batch_y, y_0, _, _ = self._make_diffusion_target(batch_y, y_T_mean, enc_out)

                e = torch.randn_like(y_0).to(self.device)
                y_t = self.model.q_sample(y_0, t, noise=e)
                output = self.model(y_t, t, enc_out)

                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.parameterization == "noise":
                    target = e
                elif self.args.parameterization == "x_start":
                    target = y_0
                else:
                    raise ValueError(f"Unknown parameterization {self.args.parameterization}")

                loss = criterion(output, target)
                loss = loss.detach().cpu()
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        if self.residual_head is not None and not self.args.freeze_residual_head_after_pretrain:
            self.residual_head.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        condition_path = os.path.join(self.args.pretrain_checkpoints, self.args.data)

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(condition_path):
            os.makedirs(condition_path)

        if getattr(self.args, "use_hrd3u", False) and self.args.pretrain_residual_head:
            self._pretrain_residual_head(train_loader, vali_loader)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        checkpoint_model = HRD3UCheckpointWrapper(self.model, self.residual_head)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if self.args.scrach_10_stop:
                if epoch == 10:
                    for param in self.cond_pred_model.parameters():
                        param.requires_grad = False

            epoch_time = time.time()
            iter_count = 0
            train_loss = []
            self.model.train()
            self.cond_pred_model.train()
            if self.residual_head is not None:
                if self.args.freeze_residual_head_after_pretrain:
                    self.residual_head.eval()
                else:
                    self.residual_head.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n].to(self.device)

                y_T_mean, enc_out = self._get_condition_outputs(batch_x, batch_y)
                batch_y, y_0, mu_r, sigma_r = self._make_diffusion_target(batch_y, y_T_mean, enc_out)

                e = torch.randn_like(y_0).to(self.device)
                y_t = self.model.q_sample(y_0, t, noise=e)
                output = self.model(y_t, t, enc_out)

                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.parameterization == "noise":
                    target = e
                elif self.args.parameterization == "x_start":
                    target = y_0
                else:
                    raise ValueError(f"Unknown parameterization {self.args.parameterization}")

                loss = (target - output).square().mean()

                # If not frozen, keep a small NLL term so the head does not drift.
                if (
                    getattr(self.args, "use_hrd3u", False)
                    and self.residual_head is not None
                    and not self.args.freeze_residual_head_after_pretrain
                    and mu_r is not None
                    and sigma_r is not None
                    and self.args.residual_locscale_weight > 0
                ):
                    raw_residual = batch_y - y_T_mean[:, -self.args.pred_len:, f_dim:]
                    locscale_loss = gaussian_nll_residual_loss(raw_residual, mu_r[:, :, f_dim:], sigma_r[:, :, f_dim:])
                    loss = loss + self.args.residual_locscale_weight * locscale_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, checkpoint_model, path)

            if math.isnan(train_loss):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model

    def calculate_batch_crps(self, pred, true):
        pool = Pool(processes=16)
        crps_values = pool.map(calculate_crps_worker, zip(pred, true))
        pool.close()
        pool.join()
        batch_crps = np.sum(crps_values)
        return batch_crps

    def _load_checkpoint(self, setting):
        ckpt = torch.load(os.path.join('checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            if self.residual_head is not None and "residual_head" in ckpt:
                self.residual_head.load_state_dict(ckpt["residual_head"])
        else:
            self.model.load_state_dict(ckpt)

    def test(self, setting, test=0):
        def exact_y_0(config, config_diff, y_tile_seq):
            y_0 = y_tile_seq.reshape(
                -1,
                int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                config.pred_len,
                config.c_out,
            )
            return y_0

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self._load_checkpoint(setting)

            condition_path = os.path.join(self.args.pretrain_checkpoints, self.args.model)
            if self.args.decomposition:
                best_condition_model_path = (
                    condition_path + '/' + str('decomposition') + '/' +
                    self.args.data_name + '/' + str(self.args.pred_len) + '/' + 'checkpoint.pth'
                )
            else:
                best_condition_model_path = (
                    condition_path + '/' + str('all') + '/' +
                    self.args.data_name + '/' + str(self.args.pred_len) + '/' + 'checkpoint.pth'
                )

            print(best_condition_model_path)
            self.cond_pred_model.load_state_dict(torch.load(best_condition_model_path, map_location=self.device))
            self.cond_pred_model = self.cond_pred_model.to(self.device)

        preds = []
        trues = []
        folder_path = '../test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()
        if self.residual_head is not None:
            self.residual_head.eval()

        self.sampler = DPMSolverSampler(self.model, self.device, self.args.parameterization)
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0.0
        sum_crps = 0.0
        sum_crps_sum = 0.0
        diag = ResidualDiagnostics()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                y_T_mean, enc_out = self._get_condition_outputs(batch_x, batch_y)

                mu_r, sigma_r = None, None
                if getattr(self.args, "use_hrd3u", False) and self.residual_head is not None and self.args.bias and not self.args.bias_y_0:
                    mu_r, sigma_r = self.residual_head(enc_out)
                    y_true = batch_y[:, -self.args.pred_len:, :]

                    diag.update(
                        y=y_true.detach(),
                        y_base=y_T_mean.detach(),
                        mu=mu_r.detach(),
                        sigma=sigma_r.detach()
                    )

                gen_y_box = []
                gen_y_bias_box = []
            
        
                for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                    repeat_n = int(
                        self.model.diffusion_config.testing.n_z_samples /
                        self.model.diffusion_config.testing.n_z_samples_depart
                    )

                    x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
                    x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                    enc_out_tile = enc_out.repeat(repeat_n, 1, 1, 1, 1) if enc_out.dim() == 4 else enc_out.repeat(repeat_n, 1, 1, 1)
                    enc_out_tile = enc_out_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                    y_T_mean_tile = y_T_mean.repeat(repeat_n, 1, 1, 1)
                    y_T_mean_tile = y_T_mean_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                    mu_r_tile, sigma_r_tile = None, None
                    if mu_r is not None and sigma_r is not None:
                        mu_r_tile = mu_r.repeat(repeat_n, 1, 1, 1)
                        mu_r_tile = mu_r_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                        sigma_r_tile = sigma_r.repeat(repeat_n, 1, 1, 1)
                        sigma_r_tile = sigma_r_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                    y_shape = (x_tile.shape[0], self.args.pred_len, x_tile.shape[-1])

                    if self.args.use_pretraining_condition:
                        enc_out_tile = y_T_mean_tile

                    if self.args.type_sampler == "none":
                        z_tile_seq = self.model.p_sample_loop(y_T_mean_tile, enc_out_tile, y_shape)
                    elif self.args.type_sampler == "DDIM":
                        z_tile_seq = self.model.fast_sample(y_T_mean_tile, enc_out_tile, y_shape, self.args.eta)
                    elif self.args.type_sampler == "DPM_solver":
                        z_tile_seq = self.sampler.sample(
                            S=self.args.DPMsolver_step,
                            conditioning=enc_out_tile,
                            shape=y_shape,
                            verbose=False,
                            unconditional_guidance_scale=1.0,
                            unconditional_conditioning=None,
                            eta=0.,
                            x_T=None,
                        )
                    else:
                        raise ValueError(f"Unknown type_sampler {self.args.type_sampler}")

                    y_tile_bias = z_tile_seq
                    if self.args.bias:
                        if getattr(self.args, "use_hrd3u", False) and mu_r_tile is not None and sigma_r_tile is not None:
                            # z -> residual -> y
                            y_tile_bias = destandardize_residual(z_tile_seq, mu_r_tile, sigma_r_tile)
                            y_tile_seq = y_T_mean_tile + y_tile_bias
                        else:
                            y_tile_seq = z_tile_seq + y_T_mean_tile
                    else:
                        y_tile_seq = z_tile_seq

                    gen_y_bias = exact_y_0(
                        config=self.model.args,
                        config_diff=self.model.diffusion_config,
                        y_tile_seq=y_tile_bias,
                    )
                    gen_y = exact_y_0(
                        config=self.model.args,
                        config_diff=self.model.diffusion_config,
                        y_tile_seq=y_tile_seq,
                    )

                    gen_y_bias_box.append(gen_y_bias.cpu().numpy())
                    gen_y_box.append(gen_y.cpu().numpy())

                outputs = np.concatenate(gen_y_box, axis=1)
                outputs_bias = np.concatenate(gen_y_bias_box, axis=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                batch_crps = self.calculate_batch_crps(pred, true)
                sum_crps += batch_crps

                pred_ns = np.mean(pred, axis=1)
                print('test2 shape:', pred_ns.shape, true.shape)
                mae, mse, rmse, mape, mspe = metric(pred_ns, true)
                print('mae_mse', mae, mse)

                total_mse += mse * pred_ns.shape[0]
                total_mae += mae * pred_ns.shape[0]
                total_samples += pred_ns.shape[0]

                preds.append(pred.sum(-1))
                trues.append(true.sum(-1))
                del outputs
                gc.collect()

                if i % 1 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60
                    ))
                    minibatch_sample_start = time.time()
                    
        if getattr(self.args, "use_hrd3u", False):
            diag.summarize(prefix=f"Residual Diagnostics on TEST: {setting}")
        print('total_samples', total_samples)
        avg_crps = sum_crps / total_samples
        mse_total = total_mse / total_samples
        mae_total = total_mae / total_samples
        print('NT metrc: CRPS:{:.4f}'.format(avg_crps))
        print('NT metrc: mse:{:.4f}, mae:{:.4f} '.format(mse_total, mae_total))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_save = np.array(preds)
        trues_save = np.array(trues)
        crps_sum = calc_quantile_CRPS_sum(preds, trues)
        print('NT metrc: CRPS_sum:{:.4f}'.format(crps_sum))

        np.save(folder_path + 'pred.npy', preds_save)
        np.save(folder_path + 'true.npy', trues_save)
