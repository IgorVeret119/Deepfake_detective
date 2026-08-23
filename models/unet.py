import torch
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from torchvision.models import vgg13, VGG13_Weights
import torch.nn.functional as F


class VGG13Encoder(nn.Module):
    """
    Энкодер на основе предобученной модели VGG13.
    Извлекает признаки из изображения, постепенно уменьшая его разрешение.
    """
    def __init__(self, num_blocks, weights=VGG13_Weights.DEFAULT):
        super().__init__()
        self.num_blocks = num_blocks

        # Загружаем предобученную VGG13 (только сверточные слои features)
        feature_extractor = vgg13(weights=weights).features

        self.blocks = nn.ModuleList()
        cur = 0
        
        # Разбиваем VGG13 на блоки, разделителем служат слои MaxPool2d
        for idx in range(self.num_blocks):
            tmp = []
            while not isinstance(feature_extractor[cur], nn.MaxPool2d):
                tmp.append(deepcopy(feature_extractor[cur]))
                cur += 1
                
            self.blocks.append(nn.Sequential(*tmp))
            cur += 1 # Пропускаем оригинальный MaxPool2d
            
        # Собственный слой пулинга
        self.MP = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            activations.append(x) # Сохраняем для skip-connections
            
            if idx != len(self.blocks) - 1:
                x = self.MP(x)
                
        return activations


class DecoderBlock(nn.Module):
    """
    Один блок декодера U-Net (с использованием конкатенации).
    """
    def __init__(self, out_channels):
        super().__init__()

        self.upconv = nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.relu = nn.ReLU()

    def forward(self, down, left):
        # Увеличиваем масштаб
        x = F.interpolate(down, scale_factor=2, mode='nearest')
        x = self.upconv(x)

        # Принудительно выравниваем пространственные размеры (H, W) декодера под энкодер
        if x.shape[2:] != left.shape[2:]:
            x = F.interpolate(x, size=left.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, left], dim=1)

        # Свёртки
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class Decoder(nn.Module):
    """
    Декодер U-Net, состоящий из нескольких DecoderBlock.
    """
    def __init__(self, num_filters, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock(num_filters * (2 ** idx)))

    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up


class UNet(nn.Module):
    """
    Финальная сборка архитектуры U-Net.
    """
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()
        
        self.encoder = VGG13Encoder(num_blocks)
        self.decoder = Decoder(64, num_blocks - 1)

        # Финальная свёртка 1x1 для получения нужного количества классов (каналов)
        self.final = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        
        return x