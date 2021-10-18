import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.backbone.get_model import get_model
from core.classifiers import FC, ODC
from core.classifiers.odc import ODCPretrain
from core.manifolds import Oblique
from core.optimizers import RiemannianAdam
from core.optimizers.rsgd import RiemannianSGD
from core.utils import setup, get_meta_data, reduce_tensor, save_state, clean_up, ensure_path, compute_loss_acc, \
    get_similarity, smooth_one_hot, cross_entropy, compute_acc
from dataloader.data_loader import get_loader





def train_pretrain(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    # if rank == 0:
    #     ensure_path(cfg.train_pretrain_log_path)
    # writer = SummaryWriter(cfg.train_pretrain_log_path)

    print(f"Running basic DDP example on rank {rank}.")
    cfg.device = f'cuda:{rank}'

    setup(rank, world_size)

    train_loader, val_loader = get_loader(cfg, pretrain=True)

    labels = torch.arange(cfg.n_way, dtype=torch.int8, device=cfg.device).repeat(
        cfg.k_shot + cfg.k_query)  # shape[75]:012340123401234...
    labels = labels.type(torch.LongTensor).cuda()

    backbone = get_model(cfg.backbone, cfg.num_class).cuda()
    # EUCLIDEAN
    # fc = FC(cfg).cuda()
    # Oblique
    manifold = Oblique()
    anchor = manifold.proj(torch.randn((cfg.T, cfg.kernels + 1, cfg.out_dim)).cuda())
    weights = manifold.proj(torch.randn((cfg.num_class, cfg.kernels + 1, cfg.out_dim)).cuda())
    fc = ODCPretrain(cfg, manifold=manifold, anchor=anchor, proto=weights).cuda()
    params = list(backbone.parameters()) + list(fc.parameters())
    # optimizer
    cfg.train_pretrain_lr = cfg.train_pretrain_lr * (world_size / 2)

    optimizer = optim.SGD(
        params, cfg.train_pretrain_lr,
        weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    # optimizer = RiemannianAdam(
    #     params, cfg.train_pretrain_lr,
    #     weight_decay=cfg.weight_decay, )

    # learning rate decay policy
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_gama, patience=cfg.lr_patient,
    #                               verbose=True)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(0.5 * cfg.train_pretrain_epochs), int(0.75 * cfg.train_pretrain_epochs)],
                            # milestones=[50, 75],
                            last_epoch=-1)

    # amp
    # backbone = convert_syncbn_model(backbone)
    # fc = convert_syncbn_model(fc)
    opt_level = cfg.opt_level
    # [backbone, fc], optimizer = amp.initialize([backbone, fc], optimizer, opt_level=opt_level)
    backbone = DistributedDataParallel(backbone, device_ids=[rank])
    fc = DistributedDataParallel(fc, device_ids=[rank])

    if cfg.train_pretrain_start_epoch:
        print('Loading Parameters from %d_model.pkl' % cfg.train_pretrain_start_epoch)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        check_point = torch.load(
            os.path.join(cfg.train_pretrain_model_path, "%d_model.pkl" % cfg.train_pretrain_start_epoch),
            map_location=map_location
            # map_location=torch.device('cpu')
        )
        backbone.load_state_dict(check_point['state_dict'])

        # scheduler.load_state_dict(check_point['scheduler'])
        optimizer.load_state_dict(check_point['optimizer'])
    # train_from_scratch
    best_acc = 0.0
    for ep in range(cfg.train_pretrain_start_epoch, cfg.train_pretrain_epochs):
        ep += 1
        loss_list = defaultdict(list)
        loss_float = defaultdict(float)
        # train
        backbone.train()
        fc.train()

        if rank == 0:
            train_pbar = tqdm(train_loader, position=rank)
        else:
            train_pbar = train_loader
        for step, data in enumerate(train_pbar):
            if cfg.use_dali:
                data = [data[0]['data'], data[0]['label']]
            if rank == 0:
                description = ["{}:{:.6f}".format(key, np.mean(loss_list[key])) for key in loss_list.keys() if
                               "train" in key]
                train_pbar.set_description(
                    f"train: rank:{rank}, {time.strftime('%H:%M:%S')} " + ','.join(description))

            # start to train
            _data = [x.to(cfg.device) for x in data]
            images, _labels = _data
            feas = []
            if cfg.mixup > 0:
                # mixup
                lam = np.random.beta(cfg.mixup, cfg.mixup)
                rand_index = torch.randperm(images.size()[0]).cuda()
                _labels = _labels.type(torch.LongTensor).cuda()
                smoothed_targets = smooth_one_hot(_labels, classes=cfg.num_class, smoothing=cfg.label_smooth)
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * images + (1 - lam) * images[rand_index]
                fea = backbone(mixed_input)
                f_mean = F.adaptive_avg_pool2d(fea, (1, 1)).squeeze().unsqueeze(1)  # num_sample, out_dim
                feas.append(f_mean)
                for i_num in range(1, cfg.kernels + 1):
                    f_max = F.adaptive_max_pool2d(fea, (i_num, i_num))  # num_sample, out_dim
                    f_mean_ = F.adaptive_avg_pool2d(f_max, (1, 1)).squeeze().unsqueeze(1)  # num_sample, out_dim
                    key = f_mean_
                    value = f_mean_
                    query = f_mean
                    f_mean_ = F.softmax(key * query / np.sqrt(key.size(-1)), -1) * value + value
                    feas.append(f_mean_)

                fea = torch.cat(feas, 1)
                logits = fc(fea)

                loss = cross_entropy(logits, target_a) * lam + cross_entropy(logits, target_b) * (1. - lam)
                acc = compute_acc(logits, target_a) * lam + compute_acc(logits, target_b) * (1. - lam)
            else:
                # forwards
                _labels = _labels.type(torch.LongTensor).cuda()
                smoothed_targets = smooth_one_hot(_labels, classes=cfg.num_class, smoothing=cfg.label_smooth)
                fea = backbone(images)
                fea = F.adaptive_avg_pool2d(fea, (1, 1)).squeeze()
                logits = fc(fea)
                loss = cross_entropy(logits, smoothed_targets)
                acc = compute_acc(logits, smoothed_targets)

            loss_float["train_loss"] = loss.item()
            loss_float["train_acc"] = acc.item()
            loss = loss / cfg.t_task

            for key in loss_float.keys():
                if "train" in key:
                    loss_list[key].append(loss_float[key])

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()

            if (step + 1) % cfg.t_task == 0:  # batch of tasks, done by accumulate gradients
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()

        # for valid
        if ep % 20 == 1:
            backbone.eval()
            fc.eval()
            if rank == 0:
                valid_pbar = tqdm(val_loader, position=0)  # only show for device_rank=0
            else:
                valid_pbar = val_loader
            for step, data in enumerate(valid_pbar):
                if cfg.use_dali:
                    data = [data[0]['data'], data[0]['label']]
                if rank == 0:
                    description = ["{}:{:.6f}".format(key, np.mean(loss_list[key])) for key in loss_list.keys() if
                                   "val" in key]
                    valid_pbar.set_description(
                        f"val: rank:{rank}, {time.strftime('%H:%M:%S')} " + ','.join(description))

                # propagation
                with torch.no_grad():
                    data = get_meta_data(data, labels, cfg)

                    _data = [x.to(cfg.device) for x in data]
                    images, _labels = _data

                    fea = backbone(images)
                    fea = F.adaptive_avg_pool2d(fea, (1, 1)).squeeze()

                    que_fea = fea[cfg.num_support:, ...].contiguous()  # (num_query, emb_size)
                    sup_fea = fea[:cfg.num_support, ...].contiguous()  # (num_support, emb_size)
                    sup_prototype = sup_fea.view(cfg.k_shot, cfg.n_way, -1).transpose(0, 1).mean(1)  # n_way, c

                    # get logits
                    logits = get_similarity(que_fea, sup_prototype, kernel=None, return_distance=False, normalize=True)
                    # calculate loss and acc
                    loss, acc = compute_loss_acc(logits, _labels[cfg.num_support:], num_class=cfg.n_way, smoothing=0)

                    loss_float["val_loss"] = reduce_tensor(loss, world_size).item()
                    loss_float["val_acc"] = reduce_tensor(acc, world_size).item()

                    for key in loss_float.keys():
                        if "val" in key:
                            loss_list[key].append(loss_float[key])

        # scheduler.step(np.mean(loss_list["val_loss"]))

        if rank == 0:
            print('epoch:{}, lr:{:.6f}'.format(ep, optimizer.param_groups[0]['lr']))
            print(["{}:{:.6f}".format(key, np.mean(loss_list[key])) for key in loss_list.keys()])

        # Model Save and Stop Criterion
        if ep % 20 == 1:
            cond1 = (np.mean(loss_list["val_acc"]) > best_acc)
            # if cond1 or cond2:
            if cond1:
                best_acc = np.mean(loss_list["val_acc"])
                best_loss = np.mean(loss_list["val_loss"])
                best_epoch = ep
                if rank == 0:
                    # print('best val loss:{:.5f}, acc:{:.5f}, save model'.format(best_loss, best_acc))
                    print('best epoch{}, val loss:{:.5f}, acc:{:.5f}, save model'.format(best_epoch, best_loss, best_acc))

                    # save model
                    # All processes should see same parameters as they all start from same
                    # random parameters and gradients are synchronized in backward passes.
                    # Therefore, saving it in one process is sufficient.
                    torch.save(save_state(backbone, optimizer, scheduler, ),
                               os.path.join(cfg.train_meta_save_path, "models", '%d_model.pkl' % ep))

                    torch.save(save_state(backbone, optimizer, scheduler, ),
                               os.path.join(cfg.train_meta_save_path,
                                            'model_best.pth.tar'))

        elif ep % 10 == 0 and rank == 0:
            print(f'save model because of {ep}')
            torch.save(save_state(backbone, optimizer, scheduler, ),
                       os.path.join(cfg.train_meta_save_path, "models", '%d_model.pkl' % ep))
        else:
            pass

    clean_up()
