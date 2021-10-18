import os.path as osp

import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', ))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))

CACHE_PATH = osp.join(ROOT_PATH, '.cache/')


def identity(x):
    return x


class FSLDateset(Dataset):
    """ Usage:
    """

    def __init__(self, setname, cfg, augment=False):
        im_size = cfg.orig_imsize
        if hasattr(cfg, 'data_size_ratio'):
            self.data_size_ratio = cfg.data_size_ratio
        else:
            self.data_size_ratio = 1

        if cfg.dataset == 'mini':
            self.IMAGE_PATH1 = osp.join(ROOT_PATH2, 'FSL_datasets/miniimagenet/images')
            self.SPLIT_PATH = osp.join(ROOT_PATH2, 'FSL_datasets/miniimagenet/split')
            self.parse_csv = self.parse_csv_mini
            csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        elif cfg.dataset == 'tiered':
            self.IMAGE_PATH1 = osp.join(ROOT_PATH2, 'FSL_datasets/tieredimagenet/images')
            self.SPLIT_PATH = osp.join(ROOT_PATH2, 'FSL_datasets/tieredimagenet/split')
            csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
            self.parse_csv = self.parse_csv_tiered
        elif cfg.dataset == 'cub':
            self.IMAGE_PATH1 = osp.join(ROOT_PATH2, 'FSL_datasets/cub')
            self.SPLIT_PATH = osp.join(ROOT_PATH2, 'FSL_datasets/cub/split')
            self.parse_csv = self.parse_csv_cub
            csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        elif cfg.dataset == 'fc100':
            self.IMAGE_PATH1 = osp.join(ROOT_PATH2, 'FSL_datasets/FC100')
            self.parse_csv = self.parse_fc100
            csv_path = osp.join(self.IMAGE_PATH1, setname)
        else:
            raise NotImplementedError()
        cache_path = osp.join(CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size))

        self.use_im_cache = (im_size != -1)  # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path)
                self.data = [resize_(Image.open(path).convert('RGB')) for path in data]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label}, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label = self.parse_csv(csv_path)

        # self.num_class = len(set(self.label))

        image_size = 84
        if augment and setname == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])

    def parse_csv_mini(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []
        lines = sorted(lines)
        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        label = np.array(label)
        data = np.array(data)
        num_classes = len(np.unique(label))
        instance_per_class = len(label) // num_classes
        select_data_length = int(self.data_size_ratio * instance_per_class)
        label = label.reshape(num_classes, instance_per_class)
        data = data.reshape(num_classes, instance_per_class)
        label = label[:, :select_data_length].reshape(-1)
        data = data[:, :select_data_length].reshape(-1)

        return data, label

    def parse_csv_cub(self, csv_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lines = sorted(lines)
        for l in tqdm(lines, ncols=64):
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(self.IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        return data, label

    def parse_csv_tiered(self, csv_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lines = sorted(lines)
        for l in tqdm(lines, ncols=64):
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(self.IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)
        # with open(csv_path, 'rb') as fo:
        #     data = pickle.load(fo)

        return data, label

    def parse_fc100(self, fc100_path):

        label_list = os.listdir(fc100_path)

        data = []
        label = []

        folders = [osp.join(fc100_path, label) for label in label_list if os.path.isdir(osp.join(fc100_path, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))

        return image, label
