import os

import torch as t
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, bias=True, sep=False):
    "3x3 convolution with SAME padding, leakyReLU"
    # if stride == 1:
    #   padding = (k - 1) // 2
    return nn.Conv2d(in_planes, out_planes, bias=bias,
                     kernel_size=3, stride=1, padding=1)

def strided_conv3x3(in_planes, out_planes, bias=True, sep=False):
    "3x3 convolution with SAME padding, leakyReLU"
    return nn.Conv2d(in_planes, out_planes, bias=bias,
                     kernel_size=3, stride=2, padding=1)

def strided_conv4x4(in_planes, out_planes, bias=True, sep=False):
    "4x4 strided convolution with SAME padding, leakyReLU"
    # e.g., input (64x64) ==> output (32x32) img (stride=2)
    return nn.Conv2d(in_planes, out_planes, bias=bias,
                     kernel_size=4, stride=2, padding=1)

def strided_conv5x5(in_planes, out_planes, bias=True, sep=False):
    "5x5 convolution with SAME padding, LeakyReLU"
    # e.g., input (64x64) ==> output (32x32) img (stride=2) 
    return nn.Conv2d(in_planes, out_planes, bias=bias,
                     kernel_size=5, stride=2, padding=2)

def conv5x5_block(in_plain, out_plain, slope=0.2):
    return nn.Sequential(
        strided_conv5x5(in_plain, out_plain),
        nn.BatchNorm2d(out_plain),
        nn.LeakyReLU(slope))

# ---------------------------------------------------------

class BasicModule(nn.Module):
    def __init__(self, path, epoch=0):
        super(BasicModule, self).__init__()
        self.path = path
        self.epoch = epoch

    def load(self):
        if os.path.isfile(self.path):
            ckpt = t.load(self.path)
            self.load_state_dict(ckpt['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.path))
            if ckpt['epoch'] is not None:
                print('   (prev_epoch: {})'.format(ckpt['epoch']))
            self.epoch = ckpt['epoch']    
        else:
            print("=> no checkpoint found at '{}'".format(self.path))

    def save(self):
        ckpt = {
            'state_dict': self.state_dict(),
            'epoch': self.epoch
        }
        t.save(ckpt, self.path)
        print("=> saved checkpoint '{}'".format(self.path))

# ---------------------------------------------------------

class BasicNet(BasicModule):
    def __init__(self, path=None):
        super(BasicNet, self).__init__(path=path)
        self.feature = nn.Sequential(    # (3, 256, 256)
            strided_conv5x5(3, 64),    # (64, 128, 128)
            nn.LeakyReLU(0.2),
            strided_conv5x5(64, 128),  # (128, 64, 64)
            nn.LeakyReLU(0.2),
            strided_conv5x5(128, 256), # (256, 32, 32)
            nn.LeakyReLU(0.2),
            strided_conv5x5(256, 512), # (512, 16, 16)
            nn.LeakyReLU(0.2),
            strided_conv5x5(512, 1024), # (1024, 8, 8)
            nn.LeakyReLU(0.2),
            strided_conv5x5(1024, 2048), # (1024, 8, 8)
            nn.LeakyReLU(0.2)
        )
        self.logit = nn.Sequential(
            nn.Linear(2048*4*4, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(-1, 2048 * 4 * 4)
        x = self.logit(x)
        return F.log_softmax(x, dim=1)


class BasicNetBN(BasicModule):
    def __init__(self, path=None):
        super(BasicNetBN, self).__init__(path=path)
        self.feature = nn.Sequential(    # (3, 256, 256)
            conv5x5_block(3, 64),    # (64, 128, 128)
            conv5x5_block(64, 128),  # (128, 64, 64)
            conv5x5_block(128, 256), # (256, 32, 32)
            conv5x5_block(256, 512), # (512, 16, 16)
            conv5x5_block(512, 1024), # (1024, 8, 8)
            conv5x5_block(1024, 2048), # (2048, 4, 4)
        )
        self.logit = nn.Sequential(
            nn.Linear(2048*4*4, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(-1, 2048 * 4 * 4)
        x = self.logit(x)
        return F.log_softmax(x, dim=1)


class BasicNetBNHalf(BasicModule):
    def __init__(self, path=None):
        super(BasicNetBNHalf, self).__init__(path=path)
        self.feature = nn.Sequential(    # (3, 256, 256)
            conv5x5_block(3, 32),    # (32, 128, 128)
            conv5x5_block(32, 64), # (64, 64, 64)
            conv5x5_block(64, 128),  # (128, 32, 32)
            conv5x5_block(128, 256), # (256, 16, 16)
            conv5x5_block(256, 512), # (512, 8, 8)
            conv5x5_block(512, 1024), # (1024, 4, 4)
        )
        self.logit = nn.Sequential(
            nn.Linear(1024*4*4, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(-1, 1024 * 4 * 4)
        x = self.logit(x)
        return F.log_softmax(x, dim=1)


class BasicNetBN(BasicModule):
    def __init__(self, path=None):
        super(BasicNetBN, self).__init__(path=path)
        self.feature = nn.Sequential(    # (3, 256, 256)
            strided_conv5x5(3, 64),    # (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            strided_conv5x5(64, 128),  # (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            strided_conv5x5(128, 256), # (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            strided_conv5x5(256, 512), # (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            strided_conv5x5(512, 1024), # (1024, 8, 8)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            strided_conv5x5(1024, 2048), # (1024, 8, 8)
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2)
        )
        self.logit = nn.Sequential(
            nn.Linear(2048*4*4, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(-1, 2048 * 4 * 4)
        x = self.logit(x)
        return F.log_softmax(x, dim=1)


class MISLNet(BasicModule):
    def __init__(self, path=None):
        super(MISLNet, self).__init__(path=path)
        self.feature = nn.Sequential(    # (3, 256, 256)
            # Lout = [( Lin + 2 * padding - kernel ) / stride] + 1
            nn.Conv2d(3, 3, kernel_size=5), #=> (3, 252, 252)
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3), #=> (96, 126, 126)
            nn.BatchNorm2d(96),
            nn.Tanh(),
            nn.MaxPool2d(3, stride=2), #=> (96, 62, 62)
            nn.Conv2d(96, 64, kernel_size=5, padding=2), #=> (64, 62, 62)
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(3, stride=2), #=> (64, 30, 30)
            nn.Conv2d(64, 64, kernel_size=5, padding=2), #=> (64, 30, 30)
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(3, stride=2), #=> (64, 14, 14)
            nn.Conv2d(64, 128, kernel_size=1), #=> (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(3, stride=2), #=> (128, 6, 6)
        )
        self.logit = nn.Sequential(
            nn.Linear(128 * 6 * 6, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(-1, 128 * 6 * 6)
        x = self.logit(x)
        return F.log_softmax(x, dim=1)


# class Discriminator(BasicModule):
#     def __init__(self, path=None):
#         assert path is not None
#         super(Discriminator, self).__init__(path)

#         self.logit = nn.Sequential(    # (3, 64, 64)
#             strided_conv4x4(3, 64),    # (64, 32, 32)
#             nn.LeakyReLU(0.2),
#             strided_conv4x4(64, 128),  # (128, 16, 16)
#             nn.LeakyReLU(0.2),
#             strided_conv4x4(128, 256), # (256, 8, 8)
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 1, kernel_size=4, padding=1), # (1, 8, 8)
#             nn.Sigmoid()
#         )

#     def forward(self, x):    # (512, 8, 8)
#         return self.logit(x) # (3, 64, 64)

# class Discriminator128(BasicModule):
#     def __init__(self, path=None):
#         assert path is not None
#         super(Discriminator128, self).__init__(path)

#         self.logit = nn.Sequential(    # (3, 128, 128)
#             strided_conv4x4(3, 64),    # (64, 64, 64) # 3 or 6
#             nn.LeakyReLU(0.2),
#             strided_conv4x4(64, 128),  # (128, 32, 32)
#             nn.LeakyReLU(0.2),
#             strided_conv4x4(128, 256), # (256, 16, 16)
#             nn.LeakyReLU(0.2),
#             strided_conv4x4(256, 512), # (512, 8, 8)
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 1, kernel_size=4, padding=1), # (1, 8, 8)
#             nn.Sigmoid()
#         )

#     def forward(self, x):    # (512, 8, 8)
#         return self.logit(x) # (3, 64, 64)
