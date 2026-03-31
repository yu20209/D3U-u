# SDRD experiment
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model9_NS_transformer.exp.exp_basic_sdrd import Exp_Basic_SDRD
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
from utils.metrics import metric

class Exp_SDRD(Exp_Basic_SDRD):
    def __init__(self, args):
        super().__init__(args)
        self.purity_weight = getattr(args, 'purity_weight', 0.0)
        self.purity_lags = getattr(args, 'purity_lags', 3)
        self.spec_weight = getattr(args, 'spec_weight', 1.0)

    def _build_model(self):
        model = diffuMTS.Model(self.args).float()
        cond_cls = self.cond_model_dict[self.args.model]
        cond_pred_model = cond_cls(self.args).float()
        return model, cond_pred_model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        params = list(self.model.parameters()) + [p for p in self.cond_pred_model.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _residual_purity_loss(self, r):
        if self.purity_weight <= 0:
            return r.new_zeros(1)
        r_centered = r - r.mean(dim=1, keepdim=True)
        acf_loss = 0.0
        for lag in range(1, self.purity_lags + 1):
            acf = (r_centered[:, :-lag, :] * r_centered[:, lag:, :]).mean()
            acf_loss = acf_loss + acf.abs()
        fft = torch.fft.rfft(r_centered, dim=1)
        power = fft.abs()
        spec_loss = power[:, 1:, :].max(dim=1).values.mean()
        return acf_loss + self.spec_weight * spec_loss

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            for batch_x, batch_y, *_ in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
                det_pred, _, enc_out = self.cond_pred_model(batch_x, None, dec_inp, None)
                y_0 = batch_y[:, -self.args.pred_len:, :] - det_pred
                t = torch.randint(0, self.model.num_timesteps, (batch_x.size(0),), device=self.device)
                noise = torch.randn_like(y_0)
                y_t = self.model.q_sample(y_0, t, noise=noise)
                output = self.model(y_t, t, enc_out)
                target = noise if self.args.parameterization == 'noise' else y_0
                loss = criterion(output, target) + self.purity_weight * self._residual_purity_loss(y_0)
                loss.backward()
                model_optim.step()
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        self.model.eval(); self.cond_pred_model.eval()
        total_mse, total_mae, total_samples = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y, *_ in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
                det_pred, _, enc_out = self.cond_pred_model(batch_x, None, dec_inp, None)
                y_shape = (batch_x.size(0), self.args.pred_len, batch_x.size(-1))
                y_sample = self.model.p_sample_loop(det_pred, enc_out, y_shape)
                y_sample = y_sample + det_pred
                pred = y_sample.cpu().numpy(); true = batch_y[:, -self.args.pred_len:, :].cpu().numpy()
                mae, mse, *_ = metric(pred, true)
                total_mse += mse * pred.shape[0]; total_mae += mae * pred.shape[0]; total_samples += pred.shape[0]
        print('Test MSE:', total_mse / total_samples)
        print('Test MAE:', total_mae / total_samples)
