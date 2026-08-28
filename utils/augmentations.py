import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms(image_size, p):
    """
    Продвинутый пайплайн аугментаций для тренировочного датасета (Deepfake Segmentation).
    p - базовая вероятность применения (например, 0.3 - 0.5)
    """
    return A.Compose([
        # ==========================================
        # 1. ГЕОМЕТРИЯ (Искажает и картинку, и маску)
        # ==========================================
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.Affine(
                scale=(0.9, 1.1), 
                translate_percent=(-0.06, 0.06), 
                rotate=(-15, 15), 
                border_mode=cv2.BORDER_REFLECT_101, # <-- Магия здесь!
                fill_mask=0, # Для маски пустоты строго заливаем нулями (черным)
                p=1.0
            ),
            # Симуляция "кривого" наложения лица (эффект кривого зеркала)
            A.GridDistortion(num_steps=5, distort_limit=0.25, border_mode=cv2.BORDER_REFLECT_101, fill_mask=0, p=1.0),
            A.OpticalDistortion(distort_limit=0.1, border_mode=cv2.BORDER_REFLECT_101, fill_mask=0, p=1.0),
            A.ElasticTransform(alpha=20, sigma=4, border_mode=cv2.BORDER_REFLECT_101, fill_mask=0, p=1.0)
        ], p=p * 0.7), # Делаем реже, чтобы не искажать слишком сильно

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
            A.ImageCompression(quality_range=(65, 95), p=1.0), # Сжатие мессенджеров
            A.Downscale(scale_range=(0.8, 0.9), p=1.0), # Симуляция апскейла (низкого разрешения)
            A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0), # Шум матрицы камеры
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0), # Смаз от движения
            A.GaussianBlur(blur_limit=(3, 7), p=1.0), # Расфокус
            A.GlassBlur(max_delta=1, iterations=1, p=1.0), # Эффект "битых" пикселей/линзы
        ], p=p * 0.6),

        # ==========================================
        # 4. УМНОЕ УДАЛЕНИЕ ЧАСТЕЙ (Cutout / CoarseDropout)
        # ==========================================
        # Вырезает от 2 до 8 черных квадратов. Заставляет модель смотреть на всё лицо, а не только на глаза.
        A.CoarseDropout(
            num_holes_range=(2, 8),        # Замена min_holes и max_holes
            hole_height_range=(16, 64),    # Замена min_height и max_height
            hole_width_range=(16, 64),     # Замена min_width и max_width
            fill=0,                        # Замена fill_value
            fill_mask=0,                   # Замена mask_fill_value
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