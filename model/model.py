import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from model.base_blocks import *

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x[:,:,:self.shape]
    
class IterativeBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel, stride, padding, activation='prelu'):
        super(IterativeBlock, self).__init__()
        bias = False
        norm = "batch"
        self.up1 = UpBlock(channels, kernel, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down1 = DownBlock(channels, kernel, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up2 = UpBlock(channels, kernel, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down2 = D_DownBlock(channels, kernel, stride, padding, 2, bias=bias, activation=activation, norm=norm)
        self.up3 = D_UpBlock(channels, kernel, stride, padding, 2, bias=bias, activation=activation, norm=norm)
        self.down3 = D_DownBlock(channels, kernel, stride, padding, 3, bias=bias, activation=activation, norm=norm)
        self.up4 = D_UpBlock(channels, kernel, stride, padding, 3, bias=bias, activation=activation, norm=norm)
        self.down4 = D_DownBlock(channels, kernel, stride, padding, 4, activation=activation)
        self.up5 = D_UpBlock(channels, kernel, stride, padding, 4, activation=activation)
        self.down5 = D_DownBlock(channels, kernel, stride, padding, 5, activation=activation)
        self.up6 = D_UpBlock(channels, kernel, stride, padding, 5, activation=activation)
        self.down6 = D_DownBlock(channels, kernel, stride, padding, 6, activation=activation)
        self.up7 = D_UpBlock(channels, kernel, stride, padding, 6, activation=activation)
        self.down7 = D_DownBlock(channels, kernel, stride, padding, 7, activation=activation)
        self.up8 = D_UpBlock(channels, kernel, stride, padding, 7, activation=activation)
        self.down8 = D_DownBlock(channels, kernel, stride, padding, 8, activation=activation)
        self.up9 = D_UpBlock(channels, kernel, stride, padding, 8, activation=activation)
        self.out_conv = ConvBlock(8 * channels, out_channels, 3, 1, 1, bias=bias, activation=activation, norm=norm)

        
    def forward(self, x):
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)
        #
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down7(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up8(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        out = self.out_conv(concat_h)

        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channnels, out_channels, stride=1, expansion=1, identity_downsample=None):
        super(ResBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channnels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.prelu(x)
        return x

class ResEncoder(nn.Module):
    def __init__(self, block, nbins: int, degree: int, latent_dim: int):
        super(ResEncoder, self).__init__()
        self.coefficient = (degree + 1) ** 2
        num_blocks = 2
        self.expansion = 1
        self.in_channels = 256
        self.conv1 = nn.Sequential(
            nn.Conv1d(nbins, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.in_channels),
            nn.PReLU(),
        )
        res_layers = []
        if degree == 19:
            strides = [2, 2, 2, 2]
            feature = 25
        elif degree == 13:
            strides = [2, 2, 2, 1]
            feature = 25
        elif degree == 9:
            strides = [2, 2, 1, 1]
            feature = 25
        elif degree == 6:
            strides = [2, 1, 1, 1]
            feature = 25
        elif degree == 4: # 32
            strides = [2, 2, 2, 1, 1]
            feature = 4
        elif degree == 3: # 48
            strides = [2, 1, 1, 1, 1]
            feature = 8
        elif degree == 2: # 72
            strides = [2, 1, 1, 1]
            feature = 5
        elif degree == 1: # 108， 216，
            strides = [1, 1, 1, 1]
            feature = 4

        res_layers.append(self._make_layer(block, 256, num_blocks))
        # print(strides)
        for stride in strides:
            res_layers.append(self._make_layer(block, 512, num_blocks, stride=stride))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc = nn.Sequential(nn.Linear(512*feature, 512),
                                nn.BatchNorm1d(512),
                                nn.PReLU(),
                                nn.Linear(512, latent_dim))
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z
    
class D_DBPN(nn.Module):
    def __init__(self, nbins, base_channels, latent_dim, max_degree):
        super(D_DBPN, self).__init__()

        max_num_coefficient = (max_degree + 1) ** 2
        kernel = 4
        stride = 2
        padding = 1
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512*16),
            nn.BatchNorm1d(512*16),
            nn.ReLU(True),
            # nn.PReLU(),
            Reshape(-1, 512, 16),
        )
        activation = 'tanh'

        self.conv0 = ConvBlock(512, base_channels, 3, 1, 1)

        # Back-projection stages
        self.up1 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        self.up2 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        self.up3 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        self.up4 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        self.up5 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        
        # Reconstruction
        self.out_conv = ConvBlock(base_channels, nbins, 3, 1, 1, activation=None)
        self.trim = Trim(max_num_coefficient)

    def forward(self, x):
        x = self.fc(x)
        x = self.conv0(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.out_conv(x)
        out = self.trim(x)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, nbins: int, in_degree: int, latent_dim: int, base_channels: int, out_degree: int=22):
        super(AutoEncoder, self).__init__()

        self.encoder = ResEncoder(ResBlock, nbins, in_degree, latent_dim)
        self.decoder = D_DBPN(nbins, base_channels=base_channels,
                              latent_dim=latent_dim, max_degree=out_degree)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight)
                    # scale = 1.0 /np.sqrt(np.prod(m.weight.shape[1:]))
                    # scale /= np.sqrt(3)
                    # nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, nbins: int) -> None:
        super(Discriminator, self).__init__()
        self.nbins = nbins

        padding = 0
        self.features = nn.Sequential(
            # input size: nbins x 529     484
            nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            # 64 x 265         242
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            # 128 x 133      121
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            # 256 x  67     61
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # 512 x 34   31
            # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # 512 x 16
        )

        # self.features = nn.Sequential(
        #     # input size: nbins x 812       841
        #     nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 406   421
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 203   211
        #     nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 102   106
        #     nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 51    53
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 26    27
        #     # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     # nn.BatchNorm1d(512),
        #     # nn.LeakyReLU(0.2, True),
        #     # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     # nn.BatchNorm1d(512),
        #     # nn.LeakyReLU(0.2, True),
        #     # nbins x 34
        # )

        self.fc = nn.Linear(512*31, 512)
        self.minibatch_discriminator = MiniBatchDiscrimination(in_features=512, out_features=100, kernel_dims=20)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 100, 512),
            nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nn.Linear(4096, 512),
            # nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.minibatch_discriminator(x)
        out = self.classifier(x)
        return out

class MiniBatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x: N x A
        # T: A x B x C
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0) # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3) # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3) # NxNxB
        expnorm = torch.exp(-norm)
        diversity = expnorm.sum(0) - 1 # NxB, subtract self distance
        if self.mean:
            diversity /= (x.size(0) - 1)
        
        x = torch.cat([x, diversity], 1)
        return x



if __name__ == '__main__':
    x = torch.randn(2, 256, 25) # 25, 9, 4
    G = AutoEncoder(nbins=256, in_degree=4, latent_dim=128, base_channels=64, out_degree=21)
    x = G(x)
    print(x.shape)
    D = Discriminator(256)
    x = D(x)
    print(x.shape)