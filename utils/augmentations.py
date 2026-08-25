import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size, p):
    """
    Продвинутый пайплайн аугментаций для тренировочного датасета (Deepfake Segmentation).
    p - базовая вероятность применения (например, 0.3 - 0.5)
    """
    return A.Compose([
        # ==========================================
        # 1. ГЕОМЕТРИЯ (Искажает и картинку, и маску)
        # ==========================================
        A.HorizontalFlip(p=p),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, border_mode=0, p=p),
        
        # Симуляция "кривого" наложения лица (эффект кривого зеркала)
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=p * 0.5), # Делаем реже, чтобы не искажать слишком сильно

        # ==========================================
        # 2. ЦВЕТ И ОСВЕЩЕНИЕ
        # ==========================================
        # Дипфейки часто имеют проблемы с балансом белого и тенями
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=p),

        # ==========================================
        # 3. ДЕГРАДАЦИЯ (Шумы, блюр, интернет-сжатие)
        # ==========================================
        A.OneOf([
            A.ImageCompression(quality_lower=50, quality_upper=95, p=1.0), # Сжатие мессенджеров
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0), # Симуляция апскейла (низкого разрешения)
        ], p=p),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0), # Шум матрицы камеры
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=p * 0.8),
        
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0), # Смаз от движения
            A.GaussianBlur(blur_limit=(3, 7), p=1.0), # Расфокус
            A.GlassBlur(max_delta=1, iterations=1, p=1.0), # Эффект "битых" пикселей/линзы
        ], p=p * 0.5),

        # ==========================================
        # 4. УМНОЕ УДАЛЕНИЕ ЧАСТЕЙ (Cutout / CoarseDropout)
        # ==========================================
        # Вырезает от 2 до 8 черных квадратов. Заставляет модель смотреть на всё лицо, а не только на глаза.
        A.CoarseDropout(
            max_holes=8, max_height=64, max_width=64,
            min_holes=2, min_height=16, min_width=16,
            fill_value=0, 
            mask_fill_value=0, # КРИТИЧНО ВАЖНО: Если мы вырезали кусок лица, маска там тоже становится нулевой!
            p=p
        ),

        # ==========================================
        # 5. ФИНАЛИЗАЦИЯ (Обязательно для нейросети)
        # ==========================================
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    """
    Пайплайн для тестового датасета.
    ЗДЕСЬ НЕТ ИСКАЖЕНИЙ! Только подготовка картинки.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])