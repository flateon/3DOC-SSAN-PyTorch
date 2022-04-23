from typing import Tuple

import torch
from torch import nn


class OctConv3d(nn.Module):
    """
        oct_type (str): The type of OctConv you'd like to use. 'first' stand for the the first Octave Convolution.
                        'last' stand for the last Octave Convolution. And 'regular' stand for the regular ones.
    """

    def __init__(self, oct_type: str, in_channels, out_channels, kernel_size: Tuple[int, int, int],
                 stride: Tuple[int, int, int] = (1, 1, 1), padding: Tuple[int, int, int] = (2, 1, 1)):
        super().__init__()
        if oct_type not in ('first', 'regular', 'last', 'last_l'):
            raise ValueError("Invalid oct type!")
        self.oct_type = oct_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.Downsample = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv3d_h2h = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.Conv3d_h2l = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.Conv3d_l2h = None if self.oct_type == 'first' else \
            nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.Conv3d_l2l = None if self.oct_type == 'first' else \
            nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, x):
        if self.oct_type == 'first':
            x_h = x
            f_h = self.Conv3d_h2h(x_h)
            f_l = self.Conv3d_h2l(self.Downsample(x_h))
            return f_h, f_l

        elif self.oct_type == 'regular':
            x_h, x_l = x
            Upsample = nn.Upsample(size=x_h.size()[2:])
            f_h = self.Conv3d_h2h(x_h) + self.Conv3d_l2h(Upsample(x_l))
            f_l = self.Conv3d_l2l(x_l) + self.Conv3d_h2l(self.Downsample(x_h))
            return f_h, f_l

        elif self.oct_type == 'last':
            x_h, x_l = x
            Upsample = nn.Upsample(size=x_h.size()[2:])
            f_h = self.Conv3d_h2h(x_h) + self.Conv3d_l2h(Upsample(x_l))
            f_l = self.Conv3d_l2l(x_l) + self.Conv3d_h2l(self.Downsample(x_h))
            return f_h + Upsample(f_l)

        elif self.oct_type == 'last_l':
            x_h, x_l = x
            Upsample = nn.Upsample(size=x_h.size()[2:])
            f_h = self.Conv3d_h2h(x_h) + self.Conv3d_l2h(Upsample(x_l))
            f_l = self.Conv3d_l2l(x_l) + self.Conv3d_h2l(self.Downsample(x_h))
            return self.Downsample(f_h) + f_l


class OctBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.BatchNorm_h = nn.BatchNorm3d(self.channels)
        self.BatchNorm_l = nn.BatchNorm3d(self.channels)

    def forward(self, x):
        x_h, x_l = x
        return self.BatchNorm_h(x_h), self.BatchNorm_l(x_l)


class OctReLu(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLu_h = nn.ReLU()
        self.ReLu_l = nn.ReLU()

    def forward(self, x):
        x_h, x_l = x
        return self.ReLu_h(x_h), self.ReLu_l(x_l)


class Conv2dBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dBnReLu, self).__init__()
        self.Conv2dBnReLu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.Conv2dBnReLu(x)


class Softmax2d(nn.Module):
    def __init__(self, start_dim):
        super(Softmax2d, self).__init__()
        self.start_dim = start_dim
        self.Softmax = nn.Softmax(self.start_dim)

    def forward(self, x):
        origin_size = x.size()
        x_flatten = torch.flatten(x, self.start_dim)
        return self.Softmax(x_flatten).view(origin_size)


class Fc(nn.Module):
    def __init__(self, mid_feature, out_feature):
        super(Fc, self).__init__()
        self.mid_feature = mid_feature
        self.out_feature = out_feature
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(self.mid_feature),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.mid_feature, self.out_feature),
        )

    def forward(self, x):
        return self.fc(x)


class _3DOC_SSAN(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self._3d_OCM = nn.Sequential(
            # Tensor[batch,1,channels,13,13]
            OctConv3d(oct_type='first', in_channels=1, out_channels=24, kernel_size=(5, 3, 3)),
            OctBatchNorm(24),
            OctReLu(),
            # (Tensor[batch,24,channels,13,13], Tensor[batch,24,channels,7,7])
            OctConv3d(oct_type='last_l', in_channels=24, out_channels=48, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            # (Tensor[batch,48,channels,7,7])
            OctConv3d(oct_type='first', in_channels=48, out_channels=24, kernel_size=(5, 3, 3)),
            OctBatchNorm(24),
            OctReLu(),
            # (Tensor[batch,24,channels,7,7], Tensor[batch,24,channels,4,4])
            OctConv3d(oct_type='last', in_channels=24, out_channels=1, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            # Tensor[batch,1,channels,7,7]
        )
        self.Softmax2d = Softmax2d(start_dim=1)
        self.spa_conv1 = Conv2dBnReLu(in_channels=self.channels, out_channels=self.channels, kernel_size=(3, 3),
                                      stride=1, padding=1)
        self.spa_conv2 = Conv2dBnReLu(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1),
                                      stride=1, padding=0)
        self.spe_conv1 = Conv2dBnReLu(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1),
                                      stride=1, padding=0)
        self.Fc_spa = Fc(mid_feature=1024, out_feature=self.num_classes)
        self.Fc_spe = Fc(mid_feature=1024, out_feature=self.num_classes)
        self.Fc_all = Fc(mid_feature=1024, out_feature=self.num_classes)

    def spatial_attention(self, x):
        # Tensor[batch,1,channels,7,7]
        x_conv1 = self.spa_conv1(x.squeeze())
        # Tensor[batch,channels,7,7]
        x_reshaped = torch.flatten(x_conv1, start_dim=2)
        # Tensor[batch,channels,49]
        x_mul = torch.matmul(torch.transpose(x_reshaped, 1, 2), x_reshaped)
        # Tensor[batch,49,49]
        x_softmax = self.Softmax2d(x_mul)
        # Tensor[batch,49,49]
        x_spa = torch.matmul(x_reshaped, x_softmax).view(x.size()) + x
        # Tensor[batch,channels,7,7]
        x_spa = self.spa_conv2(x_spa.squeeze())
        return x_spa
        # Tensor[batch,channels,7,7]

    def spectral_attention(self, x):
        # Tensor[batch,1,channels,7,7]
        x_reshaped = torch.flatten(x.squeeze(), start_dim=2)
        # Tensor[batch,channels,49]
        x_mul = torch.matmul(torch.transpose(x_reshaped, 1, 2), x_reshaped)
        # Tensor[batch,49,49]
        x_softmax = self.Softmax2d(x_mul)
        # Tensor[batch,49,49]
        x_spe = torch.matmul(x_reshaped, x_softmax).view(x.size()) + x
        # Tensor[batch,49,49]
        x_spe = self.spe_conv1(x_spe.squeeze())
        return x_spe
        # Tensor[batch,channels,7,7]

    def ssicm(self, x_spa, x_spe):
        # x_spa,x_spe: Tensor[batch,channels,7,7]
        x_spa, x_spe = torch.flatten(x_spa, start_dim=2), torch.flatten(x_spe, start_dim=2)
        # Tensor[batch,channels,49]
        x_spa_T, x_spe_T = torch.transpose(x_spa, 1, 2), torch.transpose(x_spe, 1, 2)
        spa2spe_softmax = self.Softmax2d(torch.matmul(x_spa_T, x_spe))
        spe2spa_softmax = self.Softmax2d(torch.matmul(x_spe_T, x_spa))
        c_spa2spe = torch.matmul(x_spa, spa2spe_softmax)
        c_spe2spa = torch.matmul(x_spe, spe2spa_softmax)
        t_spa = c_spe2spa + x_spa
        t_spe = c_spa2spe + x_spe
        return t_spa, t_spe, t_spa + t_spe
        # Tensor[batch,channels,49]

    def forward(self, x):
        # x: Tensor[batch,1,channels,13,13]
        x_conv3d = self._3d_OCM(x)
        # x_conv3d: Tensor[batch,1,channels,7,7]
        x_spa = self.spatial_attention(x_conv3d)
        x_spe = self.spectral_attention(x_conv3d)
        # x_spa,x_spe: Tensor[batch,channels,7,7]
        t_spa, t_spe, t_all = self.ssicm(x_spa, x_spe)
        return self.Fc_spa(t_spa), self.Fc_spe(t_spe), self.Fc_all(t_all)

# OctConv3d('first', 1, 24, (5, 3, 3))(torch.randn(10, 1, channels, 13, 13))
# x = _3DOC_SSAN()(torch.randn(10, 1, 144, 13, 13))
# print(x)
