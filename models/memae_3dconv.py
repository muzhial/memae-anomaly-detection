import torch
from torch import nn

from models import MemModule


class BasicBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=(1, 1, 1),
                 downsample=None):
        super(BasicBlock, self).__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               (3, 3, 3),
                               stride=stride,
                               padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(planes,
                               planes,
                               (3, 3, 3),
                               stride=stride,
                               padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class AutoEncoderCov3DMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov3DMem, self).__init__()

        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in,
                      feature_num_2,
                      (3, 3, 3),
                      stride=(1, 2, 2),
                      padding=(1, 1, 1)),  # (B, 96, 16, 128, 128)
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2,
                      feature_num,
                      (3, 3, 3),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1)),  # (B, 128, 8, 64, 64)
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(feature_num, feature_num),
            BasicBlock(feature_num, feature_num),
            BasicBlock(feature_num,
                       feature_num_x2,
                       downsample=nn.Conv3d(feature_num,
                                            feature_num_x2,
                                            (3, 3, 3),
                                            (1, 1, 1),
                                            (1, 1, 1))),
            nn.Conv3d(feature_num_x2,
                      feature_num_x2,
                      (3, 3, 3),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1)),  # (B, 256, 4, 32, 32)
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(feature_num_x2, feature_num_x2),
            BasicBlock(feature_num_x2, feature_num_x2),
            nn.Conv3d(feature_num_x2,
                      feature_num_x2,
                      (3, 3, 3),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1)),  # (B, 256, 2, 16, 16)
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mem_rep = MemModule(mem_dim=mem_dim,
                                 fea_dim=feature_num_x2,
                                 shrink_thres =shrink_thres)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2,
                               feature_num_x2,
                               (3, 3, 3),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2,
                               feature_num,
                               (3, 3, 3),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(feature_num, feature_num),
            nn.ConvTranspose3d(feature_num,
                               feature_num_2,
                               (3, 3, 3),
                               stride=(2, 2, 2),
                               padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(feature_num_2, feature_num_2),
            nn.ConvTranspose3d(feature_num_2,
                               self.chnum_in,
                               (3, 3, 3),
                               stride=(1, 2, 2),
                               padding=(1, 1, 1),
                               output_padding=(0, 1, 1))
        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att}
