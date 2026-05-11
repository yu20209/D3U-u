import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLocationScaleHead(nn.Module):
    """
    Lightweight residual location-scale head for HR-D3U.

    Predicts:
        mu_r    : deterministic residual correction
        sigma_r : positive heteroscedastic residual scale

    Output:
        mu_r, sigma_r: [B, pred_len, C]
    """

    def __init__(
        self,
        args,
        hidden_size=None,
        dropout=0.1,
        sigma_min=1e-3,
        sigma_max=10.0,
        mu_clip=5.0,
    ):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.c_out = args.c_out
        self.enc_in = args.enc_in
        self.use_pretraining_condition = args.use_pretraining_condition

        self.hidden_size = hidden_size or getattr(args, "residual_head_hidden", 256)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_clip = mu_clip

        if self.use_pretraining_condition:
            # condition is y_base: [B, pred_len, C]
            in_dim = self.pred_len
        else:
            # condition is enc_out: [B, C, patch_num, d_model_c].
            # D3U/PatchDN 当前默认 patch_num=12。
            in_dim = 12 * args.d_model_c

        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.pred_len),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Small last-layer init keeps the first epochs close to original D3U.
        for head in [self.mu_head, self.log_sigma_head]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _condition_to_channel_features(self, condition):
        """
        Returns:
            feat: [B*C, feature_dim]
            b, c
        """
        if condition.dim() == 4:
            # [B, C, patch_num, d_model] -> [B*C, patch_num*d_model]
            b, c, p, d = condition.shape
            return condition.reshape(b * c, p * d), b, c

        if condition.dim() == 3:
            # [B, pred_len, C] -> [B*C, pred_len]
            b, l, c = condition.shape
            return condition.permute(0, 2, 1).reshape(b * c, l), b, c

        raise ValueError(
            f"Unsupported condition shape {tuple(condition.shape)}. "
            "Expected [B,C,P,D] or [B,L,C]."
        )

    def forward(self, condition):
        feat, b, c = self._condition_to_channel_features(condition)

        # Make sure input features and head parameters live on the same device.
        feat = feat.to(next(self.parameters()).device)

        mu = self.mu_head(feat)
        if self.mu_clip is not None and self.mu_clip > 0:
            mu = self.mu_clip * torch.tanh(mu / self.mu_clip)

        raw_sigma = self.log_sigma_head(feat)
        sigma = F.softplus(raw_sigma) + self.sigma_min

        if self.sigma_max is not None and self.sigma_max > 0:
            sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)

        # [B*C, pred_len] -> [B, pred_len, C]
        device = condition.device
        mu = mu.reshape(b, c, self.pred_len).permute(0, 2, 1).contiguous().to(device)
        sigma = sigma.reshape(b, c, self.pred_len).permute(0, 2, 1).contiguous().to(device)
        return mu, sigma


def gaussian_nll_residual_loss(residual, mu, sigma):
    """
    Gaussian NLL for residual location-scale training.
    residual, mu, sigma: [B, pred_len, C]
    """
    z = (residual - mu) / sigma
    return 0.5 * (z.square() + 2.0 * torch.log(sigma)).mean()


def standardize_residual(residual, mu, sigma):
    return (residual - mu) / sigma


def destandardize_residual(z, mu, sigma):
    return mu + sigma * z
