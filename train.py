import os
import time
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Импорт конфигурации и локальных модулей
import config
from dataset import RandomBright, RandomCrop, RandomHorizontal, TrainPhotosDataset
from models.unet import (
    BiRefNetForgeryModel,
    ForgeryDetectionModel,
    UNet,
    ViTSegmentation,
)
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.cloud import CloudManager
from utils.metrics import (
    AICScoreMetric,
    BoundaryLoss,
    CombinedLoss,
    DiceLoss,
    IoUScore,
    SoftIoULoss,
)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        metric_fn,
        optimizer,
        device,
        log_dir=None,
        cloud_manager=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.device = device
        self.cloud_manager = cloud_manager

        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.global_step = 0

        self.etalone_samples = [self.val_loader.dataset[i] for i in [0, 5, 7, 9]]

    def train(self, epochs, start_epoch=0):
        print(f"Начало обучения на {epochs} эпох...")
        self._log_predictions(step=self.global_step)

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nЭпоха {epoch + 1}/{start_epoch + epochs}")
            train_loss = self._train_epoch(epoch)
            val_loss, val_metric = self._validate_epoch(epoch)

            # ==========================================
            # НОВОЕ: ЛОКАЛЬНОЕ СОХРАНЕНИЕ ВЕСОВ
            # ==========================================
            if self.writer:  # Сохраняем в папку логов
                save_dir = self.writer.log_dir
            else:
                save_dir = config.MODEL_SAVE_DIR

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")

            # Сохраняем веса модели
            torch.save(self.model.state_dict(), save_path)
            print(f"[*] Веса сохранены локально: {save_path}")
            # ==========================================

            if self.cloud_manager:
                # Обернули сохранение чекпоинта в try-except
                try:
                    self.cloud_manager.save_checkpoint(
                        epoch, self.model, self.optimizer, val_loss, val_metric
                    )
                except Exception as e:
                    print(f"[!] Не удалось отправить чекпоинт в облако: {e}")
                    print("[!] Обучение продолжается...")

            if self.writer:
                self.writer.add_scalar("Loss/test_step", val_loss, self.global_step)
                self.writer.add_scalar("Eval/test_step", val_metric, self.global_step)
                self._log_predictions(step=self.global_step)

                if self.cloud_manager:
                    # Обернули синхронизацию логов в try-except
                    try:
                        self.cloud_manager.sync_logs(self.writer.log_dir)
                    except Exception as e:
                        print(f"[!] Не удалось синхронизировать логи с облаком: {e}")

        if self.writer:
            self.writer.close()

        print("Обучение завершено.")
        return self.model

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        # tqdm с выводом Loss в реальном времени
        pbar = tqdm(self.train_loader, desc="Обучение")
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if self.writer:
                with torch.no_grad():
                    self.writer.add_scalar(
                        "Loss/train_step", loss.item(), self.global_step
                    )
                    self.global_step += 1

        return total_loss / len(self.train_loader)

    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        count = 0

        aic_metric = AICScoreMetric(threshold=config.THRESHOLD)

        with torch.no_grad():
            for X, y in tqdm(self.val_loader, desc="Валидация"):
                X, y = X.to(self.device), y.to(self.device)
                count += 1

                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                aic_metric.update(pred, y)

        final_aic, dice_pos, fpr_neg = aic_metric.compute()

        if self.writer:
            self.writer.add_scalar("Eval/Dice_pos", dice_pos, self.global_step)
            self.writer.add_scalar("Eval/FPR_neg", fpr_neg, self.global_step)

        print(
            f"Валидация завершена -> Val AIC: {final_aic:.4f} | Dice_pos: {dice_pos:.4f} | FPR_neg: {fpr_neg:.4f}"
        )

        return val_loss / count, final_aic

    def _log_predictions(self, step):
        if not self.writer:
            return

        self.model.eval()
        with torch.no_grad():
            for n, (X, y) in enumerate(self.etalone_samples, 1):
                pred = self.model(X.unsqueeze(0).to(self.device))
                pred = (torch.sigmoid(pred) > config.THRESHOLD).float()

                self.writer.add_image(
                    f"Images/example_{n}/Image", X, step, dataformats="CHW"
                )
                self.writer.add_image(
                    f"Images/example_{n}/Mask", y, step, dataformats="CHW"
                )
                self.writer.add_image(
                    f"Images/example_{n}/Result",
                    pred.cpu().squeeze(0),
                    step,
                    dataformats="CHW",
                )


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Приводит путь из CSV к абсолютному пути."""
    path_str = path.as_posix()

    if path_str.startswith("stage1/"):
        path = Path(path_str.removeprefix("stage1/"))

    if not path.is_absolute():
        path = base_dir / path

    return path


def get_data_list() -> list[tuple[Path, Path]]:
    """Загружает CSV с разметкой и формирует пары изображение-маска."""
    csv_path = Path(config.CSV_PATH)
    base_dir = Path(config.BASE_DIR)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Не найден файл разметки по пути: {csv_path}")

    df = pd.read_csv(csv_path)

    data_list = []
    for row in df.itertuples(index=False):
        row = cast(pd.Series, row)
        data_list.append(
            (
                _resolve_path(Path(row.chng_img_path), base_dir),
                _resolve_path(Path(row.gt_path), base_dir),
            )
        )
    return data_list


def drop_large_imgs(
    data_list: list[tuple[str, str]], max_size=config.MAX_INPUT_IMAGE_SIZE
):
    if max_size < 0:
        return data_list

    new_data_list = []
    for img_p, mask_p in tqdm(data_list, desc="Фильтрация больших изображений"):
        h, w = Image.open(img_p).size
        if h * w <= max_size:
            new_data_list.append((img_p, mask_p))

    return new_data_list


def main():
    print(f"Используемое устройство: {config.DEVICE}")

    data_list = get_data_list()
    data_list = drop_large_imgs(data_list, max_size=config.MAX_INPUT_IMAGE_SIZE)

    train_list, test_list = train_test_split(
        data_list, test_size=config.TEST_SPLIT, random_state=42
    )

    # Создаем объекты трансформаций
    train_transform = get_train_transforms(
        image_size=config.IMAGE_SIZE, p=config.AUGMENT_PROB
    )
    val_transform = get_val_transforms(image_size=config.IMAGE_SIZE)

    # Передаем их в датасет
    train_dataset = TrainPhotosDataset(
        data_list=train_list,
        transforms=train_transform,  # Передаем единый объект, а не список
    )

    test_dataset = TrainPhotosDataset(
        data_list=test_list,
        transforms=val_transform,  # Тестовому датасету тоже нужен transform (для Resize и нормализации)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    try:
        if config.YADISK_TOKEN:
            cloud = CloudManager(token=config.YADISK_TOKEN, remote_base_path="/IoU")
        else:
            cloud = None
    except Exception as e:
        print(f"Работаем без облака. Ошибка: {e}")
        cloud = None

    # ==========================================
    # ВЫБОР МОДЕЛИ НА ОСНОВЕ КОНФИГА
    # ==========================================
    print(f"Инициализация модели: {config.MODEL_NAME}")

    if config.MODEL_NAME == "CustomUNet":
        print("Используется кастомный U-Net (VGG13)")
        model = UNet(num_classes=config.NUM_CLASSES, num_blocks=config.NUM_BLOCKS)
        log_name = f"CustomUNet_VGG13"

    elif config.MODEL_NAME == "SMP":
        print(f"Используется современная архитектура с энкодером: {config.SMP_ENCODER}")
        model = ForgeryDetectionModel(
            encoder_name=config.SMP_ENCODER, num_classes=config.NUM_CLASSES
        )
        log_name = f"SMP_{config.SMP_ENCODER}"

    elif config.MODEL_NAME == "BiRefNet":
        print("Используется тяжелая архитектура BiRefNet")
        model = BiRefNetForgeryModel()
        log_name = "BiRefNet_SOTA"

    elif config.MODEL_NAME == "ViT_Seg":
        print("Инициализация кастомной модели ViT для сегментации...")
        model = ViTSegmentation().to(config.DEVICE)

        # Для бинарной сегментации (когда мы предсказываем маску 1 класса)
        # BCEWithLogitsLoss — это золотой стандарт.
        criterion = torch.nn.BCEWithLogitsLoss()

    else:
        raise ValueError(f"Неизвестное имя модели в конфиге: {config.MODEL_NAME}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        # Если вы уже добавили BoundaryLoss, используйте его, иначе оставьте ваш CombinedLoss
        loss_fn=CombinedLoss(
            BoundaryLoss(boundary_weight=5.0), SoftIoULoss(reduction="mean")
        ),
        metric_fn=AICScoreMetric(threshold=config.THRESHOLD),
        optimizer=optimizer,
        device=config.DEVICE,
        log_dir=os.path.join(
            config.LOG_DIR, log_name
        ),  # Папки логов теперь будут называться автоматически
        cloud_manager=cloud,
    )

    t_start = time.time()
    trainer.train(epochs=config.EPOCHS)
    print(f"Обучение завершено за {time.time() - t_start:.2f} сек.")


if __name__ == "__main__":
    main()
