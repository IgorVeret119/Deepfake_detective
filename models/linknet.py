import torch
import torch.nn.functional as F
from torch import nn

# Импортируем энкодер из соседнего модуля
from .unet import VGG13Encoder


class DecoderBlock_link(nn.Module):
    """
    Блок декодера для LinkNet.
    Использует сложение (addition) вместо конкатенации (concatenation).
    """
    def __init__(self, out_channels):
        super().__init__()

        self.upconv = nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
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

        # Сложение (ключевое отличие от U-Net)
        x += left

        # Свёртки
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class Decoder_link(nn.Module):
    """
    Декодер LinkNet, состоящий из блоков DecoderBlock_link.
    """
    def __init__(self, num_filters, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock_link(num_filters * (2 ** idx)))

    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up


class Link_net(nn.Module):
    """
    Финальная сборка архитектуры LinkNet.
    """
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()
        
        # Переиспользуем энкодер от U-Net
        self.encoder = VGG13Encoder(num_blocks)
        self.decoder = Decoder_link(64, num_blocks - 1)

        # Финальная свёртка 1x1 для попиксельной агрегации
        self.final = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        
        return x