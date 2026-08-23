import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as F_vision

# ==========================================
# Классы для аугментации данных
# ==========================================

class RandomHorizontal(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, x, mask=None):
        if torch.rand(1).item() < self.p:
            x_flipped = torch.flip(x, dims=[-1])
            mask_flipped = None if mask is None else torch.flip(mask, dims=[-1])
            return x_flipped, mask_flipped
        return x, mask

    def __repr__(self):
        return f"RandomHorizontal(p={self.p})"


class RandomCrop(object):
    def __init__(self, p, size=None, random_size=(0, 1, 0, 1)):
        self.p = p
        self.size = size
        self.random_size = random_size

    def __call__(self, x, mask=None):
        if torch.rand(1).item() < self.p:
            if self.size is not None:
                x_coord, y_coord, width, height = self.size
            else:
                h, w = x.shape[-2], x.shape[-1]
                a, b, c, d = self.random_size
                
                width = torch.randint(int(c * w), int(d * w) + 1, (1,)).item()
                height = torch.randint(int(a * h), int(b * h) + 1, (1,)).item()
                x_coord = torch.randint(0, max(1, w - width), (1,)).item()
                y_coord = torch.randint(0, max(1, h - height), (1,)).item()

            new_mask = torch.zeros_like(x)
            new_mask[..., y_coord:y_coord + height, x_coord:x_coord + width] = 1

            if mask is not None:
                new_mask_for_mask = torch.zeros_like(mask)
                y_start = max(0, y_coord - height // 10)
                y_end = min(mask.shape[-2], y_coord + height + height // 10)
                x_start = max(0, x_coord - width // 10)
                x_end = min(mask.shape[-1], x_coord + width + width // 10)
                
                new_mask_for_mask[..., y_start:y_end, x_start:x_end] = 1
                return x * new_mask, mask * new_mask_for_mask
            
            return x * new_mask, None
            
        return x, mask


class RandomBright(object):
    def __init__(self, p, value=None, mean=0, std=0.1):
        self.p = p
        self.value = value
        self.mean = mean
        self.std = std

    def __call__(self, x, mask=None):
        if torch.rand(1).item() < self.p:
            if self.value is not None:
                x = x + self.value
            else:
                noise = torch.randn(1, device=x.device) * self.std + self.mean
                x = x + noise * x.mean()
        return x, mask


class RandomBackground(object):
    def __init__(self, p, images_list, masks_list):
        self.p = p
        self.images_list = images_list
        self.masks_list = masks_list

    def __call__(self, x, mask):
        if mask is None:
            return x, mask

        if torch.rand(1).item() < self.p:
            n = torch.randint(0, len(self.images_list), (1,)).item()
            bg_image = self.images_list[n]
            bg_mask = self.masks_list[n]
            
            x_new = x.clone()
            anti_mask = (mask == 0)
            
            for c in range(min(3, x.shape[0])):
                x_new[c, anti_mask.squeeze()] = bg_image[c, anti_mask.squeeze()].to(x.dtype)
            
            bg_mask_norm = bg_mask.float()
            combined_mask = torch.max(mask.float(), bg_mask_norm)

            return x_new, combined_mask
            
        return x, mask


# ==========================================
# Класс датасета
# ==========================================

class TrainPhotosDataset(Dataset):
    """
    Датасет для загрузки изображений и масок сегментации.
    """
    def __init__(self, data_list, transforms=None):
        """
        Аргументы:
            data_list (list): Список кортежей вида (путь_к_картинке, путь_к_маске).
            transforms (list): Список функций аугментации.
        """
        self.data_list = data_list
        self.transforms = transforms if transforms is not None else []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_list[idx]

        # 1. Загрузка изображений с диска
        image = read_image(img_path)
        mask = read_image(mask_path)

        # 2. ПРИНУДИТЕЛЬНЫЙ РЕСАЙЗ (Приводим все картинки к размеру 512x512)
        # Картинку ресайзим плавно (с антиалиасингом)
        image = F_vision.resize(image, [512, 512], antialias=True)
        # Маску ресайзим методом "ближайшего соседа" (NEAREST), чтобы не размыть нули и единицы
        mask = F_vision.resize(mask, [512, 512], interpolation=F_vision.InterpolationMode.NEAREST)

        # 3. Перевод в float32 и нормализация в диапазон [0, 1]
        image = image.float() / 255.0
        mask = mask.float() / 255.0

        # 4. Нормализация ImageNet
        image = F_vision.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 5. Подготовка маски (только 1 канал, бинаризация)
        if mask.shape[0] > 1:
            mask = mask[0:1, ...] 
        mask = (mask > 0.5).float()

        # 6. Применение аугментаций
        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask