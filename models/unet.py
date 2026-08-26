import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torchvision.models import vgg13, VGG13_Weights
import segmentation_models_pytorch as smp
from transformers import AutoModelForImageSegmentation
import timm

class SRMLayer(nn.Module):
    """
    Необучаемый слой, который извлекает высокочастотный шум из картинки.
    Используются 3 классических фильтра для анализа локальных аномалий.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Горизонтальный перепад (First-order)
        f1 = [[0, 0, 0], 
              [0, -1, 1], 
              [0, 0, 0]]
              
        # 2. Горизонтальный перепад (Second-order)
        f2 = [[0, 0, 0], 
              [1, -2, 1], 
              [0, 0, 0]]
              
        # 3. Центральный лапласиан (Third-order)
        f3 = [[-1, 2, -1], 
              [2, -4, 2], 
              [-1, 2, -1]]
        
        # Собираем фильтры в тензор весов для свертки. Формат: (out_channels, in_channels, H, W)
        filters = torch.tensor([f1, f2, f3], dtype=torch.float32).unsqueeze(1)
        
        # Создаем сверточный слой
        self.noise_extractor = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
        self.noise_extractor.weight = nn.Parameter(filters, requires_grad=False) # Запрещаем сети менять эти веса
        
    def forward(self, x):
        # 1. Конвертируем RGB-изображение в оттенки серого (1 канал)
        # Формула яркости: Y = 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
        # 2. Пропускаем серое изображение через фильтры
        noise = self.noise_extractor(gray)
        
        # На выходе получаем тензор с 3 каналами чистого шума
        return noise

class ForgeryDetectionModel(nn.Module):
    """
    Универсальная модель для детекции подделок.
    Поддерживает современные энкодеры (EfficientNet, SegFormer/MiT) 
    и автоматически склеивает RGB с картами шума SRM.
    """
    def __init__(self, encoder_name="mit_b0", num_classes=1):
        super().__init__()
        
        self.srm = SRMLayer()
        
        # Создаем модель через библиотеку SMP.
        # SMP сама расширит первый слой до 6 каналов и правильно инициализирует веса!
        self.model = smp.Unet(
            encoder_name=encoder_name,     # Например: 'mit_b0' или 'efficientnet-b0'
            encoder_weights="imagenet",    # Используем предобученные веса для быстрого старта
            in_channels=6,                 # 3 RGB + 3 SRM
            classes=num_classes,
            activation=None                # Активация не нужна, так как у нас BCEWithLogitsLoss
        )

    def forward(self, x):
        # 1. Извлекаем шум
        noise = self.srm(x)
        
        # 2. Склеиваем RGB и Шум
        x_combined = torch.cat([x, noise], dim=1)
        
        # 3. Пропускаем через нейросеть
        return self.model(x_combined)

class LaptopMaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMLayer()
        
        # Создаем современный U-Net с EfficientNet-B0
        self.model = smp.Unet(
            encoder_name="efficientnet-b0", 
            encoder_weights="imagenet", 
            in_channels=6, # 3 RGB + 3 Шум
            classes=1,
            activation=None # Мы используем BCEWithLogitsLoss, поэтому активация не нужна
        )

    def forward(self, x):
        noise = self.srm(x)
        x_combined = torch.cat([x, noise], dim=1)
        return self.model(x_combined)

class VGG13Encoder(nn.Module):
    """
    Энкодер на основе предобученной модели VGG13.
    """
    def __init__(self, num_blocks, in_channels=6, weights=VGG13_Weights.DEFAULT):
        super().__init__()
        self.num_blocks = num_blocks

        # Загружаем предобученную VGG13
        feature_extractor = vgg13(weights=weights).features

        # === Модификация первого слоя VGG13 для приема 6 каналов ===
        old_conv = feature_extractor[0]
        
        # Создаем новый слой на 6 входных каналов
        new_conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        
        # Копируем готовые веса VGG13 для первых 3 каналов (RGB)
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
        # Инициализируем новые 3 канала (для SRM-шума) случайными числами
        nn.init.kaiming_normal_(new_conv.weight.data[:, 3:, :, :])
        # Копируем смещение (bias)
        new_conv.bias.data = old_conv.bias.data
        
        # Заменяем старый слой на наш новый
        feature_extractor[0] = new_conv
        # ============================================================

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
        # Это спасет от ошибки конкатенации, если размеры не кратны 32
        if x.shape[2:] != left.shape[2:]:
            x = F.interpolate(x, size=left.shape[2:], mode='bilinear', align_corners=False)

        # Склеиваем карты признаков
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
    Финальная сборка архитектуры U-Net с встроенным SRM-фильтром.
    """
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()
        
        # Инициализируем шумовой сканер
        self.srm = SRMLayer()
        
        # Энкодер теперь сразу ждет 6 каналов
        self.encoder = VGG13Encoder(num_blocks=num_blocks, in_channels=6)
        self.decoder = Decoder(64, num_blocks - 1)

        # Финальная свёртка 1x1 для получения нужного количества классов (маски)
        self.final = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # 1. Извлекаем высокочастотный шум (3 канала)
        noise = self.srm(x)
        
        # 2. Склеиваем оригинальную картинку и шум (Получаем 6 каналов)
        x_combined = torch.cat([x, noise], dim=1)
        
        # 3. Передаем все 6 каналов напрямую в энкодер
        acts = self.encoder(x_combined)
        
        # 4. Пропускаем через декодер и получаем маску
        x_out = self.decoder(acts)
        x_out = self.final(x_out)
        
        return x_out

class BiRefNetForgeryModel(nn.Module):
    """
    Интеграция SOTA модели BiRefNet с нашим SRM-фильтром.
    """
    def __init__(self):
        super().__init__()
        self.srm = SRMLayer()
        
        # Адаптер: сжимает 6 каналов (3 RGB + 3 Шум) в 3 канала, которые понимает BiRefNet
        self.input_adapter = nn.Conv2d(6, 3, kernel_size=1)
        
        # Загружаем саму модель из Hugging Face
        # trust_remote_code=True обязателен для кастомных архитектур
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", 
            trust_remote_code=True
        )

    def forward(self, x):
        noise = self.srm(x)
        x_combined = torch.cat([x, noise], dim=1)
        
        x_adapted = self.input_adapter(x_combined)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            preds = self.model(x_adapted)
        
        # 1. Распаковка словаря (если есть)
        if isinstance(preds, dict):
            preds = preds.get('preds', list(preds.values())[0])
            
        # 2. Поиск тензора с максимальным разрешением
        if isinstance(preds, (list, tuple)):
            best_pred = None
            max_pixels = -1
            
            # Функция для разворачивания любой вложенности (tuple в tuple и т.д.)
            def flatten(nested):
                for item in nested:
                    if isinstance(item, (list, tuple)):
                        yield from flatten(item)
                    elif isinstance(item, torch.Tensor):
                        yield item

            # Ищем самую большую маску
            for tensor in flatten([preds]):
                if tensor.ndim >= 2:
                    pixels = tensor.shape[-1] * tensor.shape[-2] # H * W
                    if pixels > max_pixels:
                        max_pixels = pixels
                        best_pred = tensor
                        
            final_pred = best_pred
        else:
            final_pred = preds
            
        # 3. Финальная страховка: точная подгонка размера под входную картинку x
        # Если модель вернула маску 512x512, этот код ничего не изменит.
        # Но если есть расхождение даже в 1 пиксель, он его исправит.
        if final_pred.shape[-2:] != x.shape[-2:]:
            import torch.nn.functional as F
            final_pred = F.interpolate(
                final_pred, 
                size=x.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            
        return final_pred.float()


class ViTSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # Загружаем ViT БЕЗ головы классификации (num_classes=0)
        self.encoder = timm.create_model(
            'vit_small_patch16_384.augreg_in21k_ft_in1k', 
            pretrained=True, 
            num_classes=0
        )
        
        # Простой Декодер
        # У vit_small толщина признаков (embed_dim) равна 384
        self.decoder = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # Увеличиваем в 4 раза
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # Еще в 4 раза (итого х16)
            nn.Conv2d(64, 1, kernel_size=1) # 1 канал для маски (фейк)
        )

    def forward(self, x):
        # 1. Получаем сырые токены из энкодера
        features = self.encoder.forward_features(x)
        
        # 2. Выкидываем CLS-токен (он идет под нулевым индексом)
        patch_tokens = features[:, 1:, :] 
        
        # 3. Собираем токены в 2D-сетку
        B, N, C = patch_tokens.shape
        grid_size = int(N ** 0.5) # Для 384x384 это будет 24
        
        # Меняем форму: [Batch, Channels, Height, Width]
        spatial_features = patch_tokens.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        
        # 4. Пропускаем через декодер для получения маски 384x384
        mask = self.decoder(spatial_features)
        return mask