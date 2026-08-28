import cv2
import matplotlib.pyplot as plt
import albumentations as A
# stage1/train/src/41a28965566b_000000059319.jpg
# 1. Загрузка тестовой картинки и маски (укажите свои пути)
image_path = "D:/train_stage1 (1)/stage1/train/img/8d13de81e6c8_coco_000000059319_powerpaint_realisticvision_1_Blended.jpg" 
mask_path = "D:/train_stage1 (1)/stage1/train/mask/48b2a8249536_coco_000000059319_powerpaint_realisticvision_1_Blended_mask.png"   

# Читаем через OpenCV и переводим цвета в правильный формат (RGB)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 2. Выбираем ОДНУ аугментацию для теста
# Ставим p=1.0, чтобы эффект применился со 100% вероятностью
test_transform = A.Compose([
A.CoarseDropout(
    num_holes_range=(2, 8),        # Замена min_holes и max_holes
    hole_height_range=(16, 64),    # Замена min_height и max_height
    hole_width_range=(16, 64),     # Замена min_width и max_width
    fill=0,                        # Замена fill_value
    fill_mask=0,                   # Замена mask_fill_value
    p=1.0
)
])

# 3. Применяем искажение к картинке и маске одновременно
augmented = test_transform(image=image, mask=mask)
aug_image = augmented['image']
aug_mask = augmented['mask']

# 4. Выводим результаты на экран
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Оригинал: Изображение")
axes[0, 0].axis('off')

axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title("Оригинал: Маска")
axes[0, 1].axis('off')

axes[1, 0].imshow(aug_image)
axes[1, 0].set_title("Аугментация: Изображение")
axes[1, 0].axis('off')

axes[1, 1].imshow(aug_mask, cmap='gray')
axes[1, 1].set_title("Аугментация: Маска")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()