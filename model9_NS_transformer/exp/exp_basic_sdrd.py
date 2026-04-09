import os
import torch

from model9_NS_transformer.condition_models import ns_Transformer, PatchTST_TS, SVQ
from model9_NS_transformer.condition_models.DecompExperts import Model as DecompExperts


class Exp_Basic_SDRD(object):
    def __init__(self, args):
        self.args = args
        self.cond_model_dict = {
            'DecompExperts': DecompExperts,
            'ns_Transformer': ns_Transformer,
            'PatchTST': PatchTST_TS,
            'SVQ': SVQ,
        }
        self.device = self._acquire_device()
        model, cond_pred_model = self._build_model()
        self.model = model.to(self.device)
        self.cond_pred_model = cond_pred_model.to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if torch.cuda.is_available() and self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
