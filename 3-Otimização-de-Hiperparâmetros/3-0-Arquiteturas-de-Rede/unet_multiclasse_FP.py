import torch
import torch.nn as nn

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from fuzzy_pooling import FuzzyPooling

RANDOM_SEED = 42
TEST_SIZE = 0.2

##########################################
#               U-Net D2                 #
##########################################

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, func_ativacao, tamanho_kernel=3, padding=1, use_batchnorm=True):
        super().__init__()

        camadas = [
            nn.Conv2d(
                in_ch, 
                out_ch, 
                tamanho_kernel, 
                padding=padding
            )
        ]

        if use_batchnorm:
            camadas.append(nn.BatchNorm2d(out_ch))

        camadas.append(
            func_ativacao
        )

        camadas.append(
            nn.Conv2d(
                out_ch, 
                out_ch, 
                tamanho_kernel, 
                padding=padding
            )
        )

        if use_batchnorm:
            camadas.append(nn.BatchNorm2d(out_ch))

        camadas.append(
            func_ativacao
        )

        self.conv = nn.Sequential(*camadas)

    def forward(self, x):
        return self.conv(x)


class UNet_MulticlasseD2_FP(nn.Module):
    def __init__(
        self,
        tamanho_kernel,  
        func_ativacao, 
        use_batchnorm = True,
        n_channels=1,
        n_classes=3
    ):
        super().__init__()

        padding = tamanho_kernel // 2

        self.down1 = DoubleConv(
            n_channels, 
            16, 
            func_ativacao = func_ativacao,
            tamanho_kernel=tamanho_kernel, 
            padding=padding, 
            use_batchnorm=True 
        )

        self.pool1 = nn.FuzzyPooling()

        self.down2 = DoubleConv(
            16,
            32,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.pool2 = nn.FuzzyPooling()

        self.down3 = DoubleConv(
            32,
            64,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.pool3 = nn.FuzzyPooling()

        self.middle = DoubleConv(
            64,
            128,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.conv3 = DoubleConv(
            128,
            64,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.conv2 = DoubleConv(
            64,
            32,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            2,
            stride=2
        )

        self.conv1 = DoubleConv(
            32,
            16,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.outc = nn.Conv2d(
            16,
            n_classes,
            1
        )

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        m = self.middle(self.pool3(d3))

        u3 = self.up3(m)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.outc(u1)

##########################################
#               U-Net D3                 #
##########################################

class TripleConv(nn.Module):

    def __init__(self, in_ch, out_ch, func_ativacao, tamanho_kernel=3, padding=1, use_batchnorm=True):
        super().__init__()

        # Camada 1
        camadas = [
            nn.Conv2d(
                in_ch, 
                out_ch, 
                tamanho_kernel, 
                padding=padding
            )
        ]

        if use_batchnorm:
            camadas.append(nn.BatchNorm2d(out_ch))

        camadas.append(
            func_ativacao
        )

        # Camada 2
        camadas.append(
            nn.Conv2d(
                out_ch, 
                out_ch, 
                tamanho_kernel, 
                padding=padding
            )
        )

        if use_batchnorm:
            camadas.append(nn.BatchNorm2d(out_ch))

        camadas.append(
            func_ativacao
        )

        # Camada 3
        camadas.append(
            nn.Conv2d(
                out_ch, 
                out_ch, 
                tamanho_kernel, 
                padding=padding
            )
        )

        if use_batchnorm:
            camadas.append(nn.BatchNorm2d(out_ch))

        camadas.append(
            func_ativacao
        )

        self.conv = nn.Sequential(*camadas)

    def forward(self, x):
        return self.conv(x)


class UNet_MulticlasseD3_FP(nn.Module):
    def __init__(
        self,
        tamanho_kernel,  
        func_ativacao, 
        use_batchnorm = True,
        n_channels=1,
        n_classes=3
    ):
        super().__init__()

        padding = tamanho_kernel // 2

        self.down1 = TripleConv(
            n_channels, 
            16, 
            func_ativacao = func_ativacao,
            tamanho_kernel=tamanho_kernel, 
            padding=padding, 
            use_batchnorm=True 
        )

        self.pool1 = nn.FuzzyPooling()

        self.down2 = TripleConv(
            16,
            32,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.pool2 = nn.FuzzyPooling()

        self.down3 = TripleConv(
            32,
            64,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.pool3 = nn.FuzzyPooling()

        self.middle = TripleConv(
            64,
            128,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.conv3 = TripleConv(
            128,
            64,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.conv2 = TripleConv(
            64,
            32,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            2,
            stride=2
        )

        self.conv1 = TripleConv(
            32,
            16,
            func_ativacao,
            tamanho_kernel,
            padding,
            use_batchnorm
        )

        self.outc = nn.Conv2d(
            16,
            n_classes,
            1
        )

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        m = self.middle(self.pool3(d3))

        u3 = self.up3(m)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.outc(u1)