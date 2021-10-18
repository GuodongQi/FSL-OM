### dropout has been removed in this code. original code had dropout#####
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

act = torch.nn.ReLU()

import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    if torch.cuda.is_available():
        y_onehot = y_onehot.cuda()

    y_onehot.zero_()
    x = inp.type(torch.LongTensor)
    if torch.cuda.is_available():
        x = x.cuda()

    x = torch.unsqueeze(x, 1)
    y_onehot.scatter_(1, x, 1)

    return Variable(y_onehot, requires_grad=False)
    # return y_onehot


def mixup_data(x, y, lam):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=200, loss_type='dist', per_img_std=False, stride=1,
                 dropRate=0.5):
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        # if loss_type == 'softmax':
        #     self.linear = nn.Linear(nChannels[3], int(num_classes))
        #     self.linear.bias.data.fill_(0)
        # else:
        #     self.linear = backbone.distLinear(nChannels[3], int(num_classes))
        self.num_classes = num_classes
        if flatten:
            self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, target=None, mixup=False, mixup_hidden=True, mixup_alpha=None, lam=0.4):
        if target is not None:
            if mixup_hidden:
                layer_mix = random.randint(0, 3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None

            out = x

            target_a = target_b = target

            if layer_mix == 0:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.conv1(out)
            out = self.block1(out)

            if layer_mix == 1:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.block2(out)

            if layer_mix == 2:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.block3(out)
            if layer_mix == 3:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)

            return out, out1, target_a, target_b
        else:
            out = x
            out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            # out = F.avg_pool2d(out, out.size()[2:])
            # out = out.view(out.size(0), -1)
            # out1 = self.linear(out)
            # return out, out1
            return out


def wrn_mixup(num_classes=10, drop_rate=0.5, loss_type='dist'):
    model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, loss_type=loss_type, per_img_std=False,
                       stride=1, dropRate=drop_rate)
    return model