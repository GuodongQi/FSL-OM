import argparse
import os
import time

import torch

from core.utils import ensure_path, set_gpu, run_model, str2bool
from eval import fine_tune
from train_pretrain import train_pretrain

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, default='resnet18', choices=["resnet18", "wideres"])

# dataset
parser.add_argument('--dataset', type=str, default='mini', choices=["mini", "tiered", 'cub', 'mini2cub', 'fc100'])
parser.add_argument('--num_workers', type=int, default=8, help="dataloader num_works")
parser.add_argument('--val_set', type=str, default='val', choices=['val', 'test'], help='the set for validation')

# FSL meta setting
parser.add_argument('--t_task', type=int, default=1, help="number of batch tasks")
parser.add_argument('--n_way', type=int, default=5, help="N-way")
parser.add_argument('--k_shot', type=int, default=5, help="K-shot")
parser.add_argument('--k_query', type=int, default=15, help="K-query")
parser.add_argument('--train_episodes', type=int, default=900, help="train episodes")

parser.add_argument('--skip_pretrain', type=str2bool, default=True, help="whether to skip pretrain")

# optimization params
parser.add_argument('--train_pretrain_lr', type=float, default=1e-1, metavar='LR', help="base learning rate")
parser.add_argument('--lr_weights', type=float, default=1e-1, metavar='LR', help="fine tune learning rate")
parser.add_argument('--lr_anchors', type=float, default=1e-1, metavar='LR', help="fine tune learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum")

# mixups
parser.add_argument('--mixup', type=float, default=0.1, help="momentum")

# learning rate decay policy
parser.add_argument('--lr_gama', type=float, default=0.1, metavar='GAMMA',
                    help="decay rate")
parser.add_argument('--lr_patient', type=int, default=20, metavar='LR Patient ',
                    help="learning schedule patients")

# total epoch, save and restore params
parser.add_argument('--train_pretrain_epochs', type=int, default=200, help="epoch")
parser.add_argument('--train_pretrain_start_epoch', type=int, default=0, help="epoch to restore params")
parser.add_argument('--save_path', type=str, default="checkpoints")

parser.add_argument('--train_meta_epochs', type=int, default=40, help="epoch")
parser.add_argument('--train_meta_start_epoch', type=int, default=0, help="epoch to restore params")
parser.add_argument('--train_meta_step', type=int, default=50, help="scheduler step")


# APEX
parser.add_argument('--gpu', default='4,5,6')
parser.add_argument('--opt_level', type=str, default='O0')

parser.add_argument('--use_dali', type=str2bool, default=False, help="whether to use dali")
parser.add_argument('--port', type=int, default=23333, help="port")
parser.add_argument('--seed', type=int, default=12667, help="seed")

# experiments
parser.add_argument('--T', type=int, default=14, help=" number of anchors - 1")
parser.add_argument('--kernels', type=int, default=11, help="number of kernels")


args = parser.parse_args()
args.orig_imsize = -1

args.val_episodes = 900 if args.k_shot == 5 else 4000

args.m = 0
if args.backbone == 'resnet18':
    args.out_dim = 512
    args.batch_size = 256
else:
    args.out_dim = 640
    args.batch_size = 128

args.label_smooth = 0.1

args.w1 = 1
args.w2 = 10
args.w3 = 1
args.scale_factor = 15

if args.dataset == 'mini':
    args.num_class = 64
elif args.dataset == 'tiered':
    args.num_class = 351
elif args.dataset == 'mini2cub':
    args.num_class = 64
elif args.dataset == 'cub':
    args.num_class = 100
elif args.dataset == 'fc100':
    args.num_class = 60
    # args.batch_size = int(args.batch_size / 1.5)

args.num_support = args.n_way * args.k_shot
args.num_query = args.n_way * args.k_query
args.num_samples = args.num_support + args.num_query

# set gpu
set_gpu(args.gpu)

# model path, log path
pwd = os.getcwd()

local_time = time.strftime('%Y-%m-%d-%H-%M')

# path
args.train_meta_save_path = os.path.join(pwd, args.save_path, args.dataset, 'softmax', args.backbone)
args.train_pretrain_model_path = os.path.join(args.train_meta_save_path, "models")
args.train_pretrain_log_path = os.path.join(args.train_meta_save_path, "logs", "%s" % local_time)
args.train_pretrain_best_model = os.path.join(args.train_meta_save_path, 'model_best.pth.tar')

ensure_path(args.train_meta_save_path)
ensure_path(args.train_pretrain_model_path)
# ensure_path(args.train_pretrain_log_path)

torch.backends.cudnn.benchmark = True


def main(rank, world_size, cfg):
    cfg.world_size = world_size
    if not cfg.skip_pretrain:
        print("Start training pretrain")
        train_pretrain(rank, world_size, cfg)
        print('pretrain finished! ')
        time.sleep(5)
    print("Start training meta")
    cfg.val_set = 'test'
    fine_tune(rank, world_size, cfg)


if __name__ == '__main__':
    world_size_ = torch.cuda.device_count()
    run_model(main, world_size_, args)
