import numpy as np
import matplotlib.pyplot as plt

def show_idx_image(dataset, idx):
    """
    Отображает изображение и соответствующую ему маску из датасета по заданному индексу.
    Автоматически выполняет денормализацию ImageNet.

    Аргументы:
        dataset: Объект датасета (возвращающий image_tensor, mask_tensor).
        idx (int): Индекс элемента в датасете.
    """
    # Получаем тензоры из датасета
    image, mask = dataset[idx]

    # Перевод из (C, H, W) в (H, W, C) для matplotlib
    image = image.permute(1, 2, 0).numpy()
    
    # Денормализация (ImageNet mean и std)
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    image = (image * std) + mean
    
    # Обрезаем значения, чтобы не было предупреждений от matplotlib
    image = np.clip(image, 0, 1)

    # Убираем лишнее измерение каналов у маски (из 1xHxW делаем HxW)
    mask = mask.squeeze(0).numpy()

    # Создаем холст
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Отрисовка
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].set_axis_off()
    
    axes[1].imshow(mask, cmap='gray') # Маску лучше выводить в ЧБ
    axes[1].set_title("Mask")
    axes[1].set_axis_off()

    fig.tight_layout()
    plt.show()