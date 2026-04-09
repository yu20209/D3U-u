import torch
from torch import nn

from layers.Decompose import series_decomp, FourierLayer
from model9_NS_transformer.condition_models import ns_Transformer, PatchTST_TS, SVQ


class SharedTemporalHead(nn.Module):
    """Project a temporal component from lookback window to forecast horizon.

    Input/Output shape: [B, L, C] -> [B, H, C]
    The projection is shared across variables to keep the trend/seasonal experts simple.
    """

    def __init__(self, seq_len, pred_len, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class Model(nn.Module):
    """Structure-decomposed deterministic experts.

    The wrapper builds three deterministic experts:
    1. trend expert    : moving-average low-frequency projection
    2. seasonal expert : Fourier seasonality projection
    3. structure expert: existing D3U conditioning backbone (SVQ / PatchTST / ns_Transformer)

    The summed deterministic forecast is used as the residual anchor for diffusion.
    The structure expert still provides patch/token features for PatchDN.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.structure_backbone_name = getattr(configs, 'structure_backbone', 'SVQ')
        self.structure_on_residual = getattr(configs, 'structure_on_residual', True)
        self.trend_weight = getattr(configs, 'trend_weight', 1.0)
        self.seasonal_weight = getattr(configs, 'seasonal_weight', 1.0)
        self.structure_weight = getattr(configs, 'structure_weight', 1.0)

        backbone_dict = {
            'ns_Transformer': ns_Transformer,
            'PatchTST': PatchTST_TS,
            'SVQ': SVQ,
        }
        if self.structure_backbone_name not in backbone_dict:
            raise ValueError(f'Unsupported structure backbone: {self.structure_backbone_name}')

        self.decomp = series_decomp(kernel_size=getattr(configs, 'trend_kernel', configs.kernel_size))
        self.seasonal = FourierLayer(d_model=configs.d_model_c, factor=configs.fourier_factor)
        self.trend_head = SharedTemporalHead(configs.seq_len, configs.pred_len, dropout=configs.dropout)
        self.seasonal_head = SharedTemporalHead(configs.seq_len, configs.pred_len, dropout=configs.dropout)
        self.structure_backbone = backbone_dict[self.structure_backbone_name].Model(configs)

    def decompose_history(self, x):
        residual, trend = self.decomp(x)
        seasonal = self.seasonal(residual)
        structure_input = x - trend - seasonal if self.structure_on_residual else x
        return trend, seasonal, structure_input

    def deterministic_forecast(self, x, batch_x_mark, dec_inp, batch_y_mark):
        trend, seasonal, structure_input = self.decompose_history(x)
        trend_pred = self.trend_head(trend)
        seasonal_pred = self.seasonal_head(seasonal)
        structure_pred, _, structure_feat = self.structure_backbone(
            structure_input, batch_x_mark, dec_inp, batch_y_mark
        )
        det_pred = (
            self.trend_weight * trend_pred
            + self.seasonal_weight * seasonal_pred
            + self.structure_weight * structure_pred
        )
        return det_pred, structure_feat, {
            'trend_pred': trend_pred,
            'seasonal_pred': seasonal_pred,
            'structure_pred': structure_pred,
            'structure_input': structure_input,
        }

    def load_structure_backbone(self, state_dict, strict=False):
        return self.structure_backbone.load_state_dict(state_dict, strict=strict)

    def freeze_structure_backbone(self):
        for param in self.structure_backbone.parameters():
            param.requires_grad = False

    def unfreeze_structure_backbone(self):
        for param in self.structure_backbone.parameters():
            param.requires_grad = True

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark, vq_details=True):
        det_pred, structure_feat, _ = self.deterministic_forecast(x, batch_x_mark, dec_inp, batch_y_mark)
        zero = det_pred.new_zeros(1)
        return det_pred, zero, structure_feat
