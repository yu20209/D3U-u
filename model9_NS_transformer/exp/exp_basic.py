import os
import torch
import numpy as np
from model9_NS_transformer.condition_models import ns_Transformer, PatchTST_TS, SVQ

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.cond_model_dict = {
            'ns_Transformer': ns_Transformer,
            'PatchTST': PatchTST_TS,
            'SVQ': SVQ
        }
        self.device = self._acquire_device()
        model, cond_pred_model = self._build_model()
        self.model = model.to(self.device)
        self.cond_pred_model = cond_pred_model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None, None, None

    def _acquire_device(self):
        """
        Device selection priority:
        1) CUDA (if available and args.use_gpu == True)
        2) MPS (only if available; mainly for macOS)
        3) CPU fallback
        """
        # 1) CUDA
        if torch.cuda.is_available() and getattr(self.args, "use_gpu", False):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
            return device

        # 2) MPS (macOS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print('Use MPS')
            return device

        # 3) CPU fallback
        device = torch.device('cpu')
        print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
