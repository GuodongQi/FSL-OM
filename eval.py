import time
from collections import defaultdict

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from tqdm import tqdm

from core.backbone.get_model import get_model
from core.classifiers import ODC
from core.manifolds import Oblique, Euclidean
from core.optimizers import RiemannianAdam
from core.utils import setup, get_meta_data, clean_up, compute_loss_acc, set_seed, reduce_tensor, smooth_one_hot
from dataloader.data_loader import get_loader


def fine_tune(rank, world_size, cfg):
    set_seed(cfg.seed, rank)
    print_rank = 0

    torch.cuda.set_device(rank)
    print(f"Running basic DDP example on rank {rank}.")
    cfg.device = f'cuda:{rank}'
    setup(rank, world_size, cfg.port)

    train_loader, val_loader = get_loader(cfg)

    labels = torch.arange(cfg.n_way, dtype=torch.long, device=cfg.device).repeat(
        cfg.k_shot + cfg.k_query)  # shape[75]:012340123401234...

    #################################################################
    # backbone and load backbone state_dict
    #################################################################
    backbone = get_model(cfg.backbone, cfg.num_class).cuda()
    backbone = DistributedDataParallel(backbone, device_ids=[rank])

    print(f'Loading Parameters from pretrain model: {cfg.train_pretrain_best_model}')
    model_dict = backbone.state_dict()
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    map_location = 'cpu'
    pretrained_dict = torch.load(cfg.train_pretrain_best_model,
                                 map_location=map_location
                                 # map_location=torch.device('cpu')
                                 )['state_dict']

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    backbone.load_state_dict(model_dict)

    # gcn = DistributedDataParallel(backbone)

    # Train and validation
    best_acc_list = []
    loss_float = defaultdict(float)
    if rank == print_rank:
        valid_pbar = tqdm(val_loader, position=0,
                          # bar_format="{desc:9.9s}: {percentage:0.1f}%|{bar}{r_bar}"
                          )  # only show for device_rank=0
    else:
        valid_pbar = val_loader

    val_acc = 0
    pm = 0

    for step, data in enumerate(valid_pbar):
        if rank == print_rank:
            description = ["{}:{:.6f}".format(key, loss_float[key]) for key in loss_float]
            valid_pbar.set_description(
                f"val: rank:{rank}, {time.strftime('%H:%M:%S')} " + ','.join(description))

        if cfg.use_dali:
            data = [data[0]['data'], data[0]['label']]
        data = get_meta_data(data, labels, cfg)
        _data = [x.to(cfg.device) for x in data]
        images, _labels = _data
        backbone.eval()

        feas = []
        anchor = []
        with torch.no_grad():
            fea = backbone(images)

            # RSMA
            f_mean = F.adaptive_avg_pool2d(fea, (1, 1)).squeeze().unsqueeze(1)  # num_sample, out_dim
            feas.append(f_mean)
            for i_num in range(1, cfg.kernels + 1):
                f_max = F.adaptive_max_pool2d(fea, (i_num, i_num))  # num_sample, i, i out_dim
                key = f_max.permute(0, 2, 3, 1)
                value = f_max.permute(0, 2, 3, 1)
                query = f_mean.unsqueeze(1)
                f_m = F.softmax(key * query / np.sqrt(key.size(-1)), -1) * value + value
                f_mean_ = F.adaptive_avg_pool2d(f_m.permute(0, 3, 2, 1), (1, 1)).squeeze().unsqueeze(1)  # num_sample, out_dim
                feas.append(f_mean_)

            fea = torch.cat(feas, 1)

            # initialization for weights
            sup_prototype = fea[:cfg.num_support, ...] \
                .contiguous().view(cfg.k_shot, cfg.n_way, fea.size(-2), fea.size(-1)).transpose(0, 1).mean(
                1)  # n_way, num, out_dim//num

            # sup_prototype = torch.randn_like(sup_prototype)

            # initialization for anchors
            anchor.append(fea[cfg.num_support:, ...].sum(0, keepdim=True) / cfg.num_support)  # T=0
            for t in range(1, cfg.T + 1):
                anchor.append(
                    ((cfg.T - t) * fea[cfg.num_support:, ...].sum(0, keepdim=True) +
                     t * fea[:cfg.num_support, ...].sum(0, keepdim=True)) /
                    ((cfg.T - t) * cfg.num_support + t * cfg.num_query))

            if len(anchor) > 1:
                anchor = torch.cat(anchor, 0)
            else:
                anchor = anchor[0]

        mani = "Euclidean"
        # mani = "oblique"
        if mani == "oblique":
            manifold = Oblique()
        else:
            manifold = Euclidean()

        projection = manifold.proj(fea)  # project to manifold
        sup_prototype = manifold.proj(sup_prototype)  # project to manifold
        anchor = manifold.proj(anchor)

        que_proj = projection[cfg.num_support:, ...].contiguous()  # (num_query, out_dim, out_dim )
        sup_proj = projection[:cfg.num_support, ...].contiguous()  # (num_support, out_dim, out_dim)

        sup_fea = sup_proj.clone().detach()
        que_fea = que_proj.clone().detach()
        sup_labels = _labels[:cfg.num_support].contiguous().clone().detach()
        que_labels = _labels[cfg.num_support:].contiguous().clone().detach()

        # fineTune
        #################################################################

        # init weights and layers
        fc = ODC(cfg, manifold=manifold, anchor=anchor, proto=sup_prototype).cuda()

        acc = []
        # optimizer = RiemannianAdam(
        #     [{'params': fc.parameters(), 'lr': cfg.train_meta_lr}],
        #     # weight_decay=cfg.weight_decay,
        # )
        # scheduler = StepLR(optimizer, cfg.train_meta_step, gamma=0.1, last_epoch=-1)

        optimizer = RiemannianAdam(
            [{'params': fc.proto, 'lr': cfg.lr_weights}, {'params': fc.anchor, 'lr': cfg.lr_anchors}],
            # weight_decay=1e-4,
        )
        # scheduler = StepLR(optimizer, 400, gamma=0.5, last_epoch=-1)
        scheduler = CosineAnnealingLR(optimizer, 0.1 * cfg.train_meta_epochs, eta_min=1e-9)

        # optimizer = RiemannianSGD(
        #     fc.parameters(), lr=1e-3,
        #     weight_decay=cfg.weight_decay, momentum=0.99)
        # scheduler = StepLR(optimizer, 40, gamma=0.1, last_epoch=-1)
        # scheduler = MultiStepLR(optimizer,
        #                         milestones=[int(0.5 * cfg.train_meta_epochs),
        #                                     int(0.75 * cfg.train_meta_epochs)],
        #                         # milestones=[50, 75],
        #                         last_epoch=-1)

        # amp
        opt_level = cfg.opt_level

        # fc, optimizer = amp.initialize(fc, optimizer, opt_level=opt_level, verbosity=0)
        loss_weight = [cfg.w1, cfg.w2, cfg.w3]
        acc_max = 0
        # for ep in range(0, cfg.train_meta_epochs):
        for ep in range(0, cfg.train_meta_epochs):
            ep += 1
            fc.train()

            logits = fc(torch.cat([sup_fea, que_fea], 0))

            sup_logits = logits[:cfg.num_support, ...] - 0 * smooth_one_hot(sup_labels, classes=cfg.n_way, smoothing=0)
            que_logits = logits[cfg.num_support:, ...]

            # compute loss and acc
            sup_loss, sup_acc = compute_loss_acc(sup_logits, sup_labels, num_class=cfg.n_way, smoothing=0, )

            if cfg.T == 0:
                loss = loss_weight[0] * sup_loss
            else:
                que_probs = que_logits.softmax(-1)
                que_cond_ent = -(que_probs * torch.log(que_probs + 1e-12)).sum(-1).mean(0)
                que_ent = -(que_probs.mean(0) * torch.log(que_probs.mean(0))).sum(-1)
                loss = loss_weight[0] * sup_loss - (loss_weight[1] * que_ent - loss_weight[2] * que_cond_ent)

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # fc.eval()
            # logits = fc(torch.cat([sup_fea, que_fea], 0))
            # que_logits = logits[cfg.num_support:, ...]
            que_loss, que_acc = compute_loss_acc(que_logits, que_labels, num_class=cfg.n_way, smoothing=0)
            if rank == print_rank and ep % 1090 == 0:
                print(ep, f"{optimizer.param_groups[0]['lr']:.5f}", sup_loss.item(), sup_acc.item(),
                      (loss_weight[1] * que_ent - loss_weight[2] * que_cond_ent).item(), que_loss.item(),
                      que_acc.item())
            acc.append(que_acc)



        acc = torch.stack(acc, 0)[-1]
        gather_t = [torch.ones_like(acc) for _ in range(world_size)]
        torch.distributed.all_gather(gather_t, acc.clone().detach())

        best_acc_list_ = [acc_item.item() for acc_item in gather_t]
        best_acc_list = best_acc_list + best_acc_list_

        cur_acc = np.mean(best_acc_list_) + 0
        val_acc = np.mean(best_acc_list) + 0
        std = np.std(best_acc_list)
        pm = 1.96 * (std / np.sqrt(len(best_acc_list)))

        loss_float['cur_acc'] = cur_acc
        loss_float['val_acc'] = val_acc
        loss_float['pm'] = pm

 
    if rank == print_rank:
        print(f'val_acc:{val_acc * 100:.3f}, pm:{pm * 100:.3f}')
        if hasattr(cfg, 'trial'):
            print(f'w1:{cfg.w1:.6f}, w2{cfg.w2:.6f}, w3{cfg.w3:.6f}, scale_factor:{cfg.scale_factor:.6f}')
    clean_up()

