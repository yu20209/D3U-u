import argparse
import torch
import random
import numpy as np

from model9_NS_transformer.exp.exp_sdrd import Exp_SDRD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDRD: Structure-Decomposed Residual Diffusion')

    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--model', type=str, default='DecompExperts')
    parser.add_argument('--structure_backbone', type=str, default='SVQ')

    parser.add_argument('--data_name', type=str, default='ETTh2')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv')
    parser.add_argument('--features', type=str, default='M')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=192)

    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)

    parser.add_argument('--trend_kernel', type=int, default=15)
    parser.add_argument('--purity_weight', type=float, default=0.0)
    parser.add_argument('--purity_lags', type=int, default=3)

    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    Exp = Exp_SDRD
    exp = Exp(args)

    if args.is_training:
        exp.train('sdrd_exp')
        exp.test('sdrd_exp')
    else:
        exp.test('sdrd_exp', test=1)
