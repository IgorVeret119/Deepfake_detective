import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Импорты локальных модулей
from models import UNet, Link_net
from utils.metrics import DiceLoss, IoUScore, SoftIoULoss, CombinedLoss, AICScoreMetric
from utils.cloud import CloudManager
from dataset import TrainPhotosDataset, RandomHorizontal, RandomCrop, RandomBright


class Trainer:
    """
    Универсальный класс для обучения моделей сегментации.
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, metric_fn, 
                 optimizer, device, log_dir=None, cloud_manager=None):
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
            
            if self.cloud_manager:
                self.cloud_manager.save_checkpoint(
                    epoch, self.model, self.optimizer, val_loss, val_metric
                )
                
            if self.writer:
                self.writer.add_scalar('Loss/test_step', val_loss, self.global_step)
                self.writer.add_scalar('Eval/test_step', val_metric, self.global_step)
                self._log_predictions(step=self.global_step)
                
                if self.cloud_manager:
                    self.cloud_manager.sync_logs(self.writer.log_dir)

        if self.writer:
            self.writer.close()
            
        print("Обучение завершено.")
        return self.model

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for X, y in tqdm(self.train_loader, desc="Обучение"):
            X, y = X.to(self.device), y.to(self.device)
            
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.writer:
                with torch.no_grad():
                    self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                    self.global_step += 1
                    
        return total_loss / len(self.train_loader)

    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        count = 0
        
        # Используем кастомную метрику AIC Score
        aic_metric = AICScoreMetric(threshold=0.8)
        
        with torch.no_grad():
            for X, y in tqdm(self.val_loader, desc="Валидация"):
                X, y = X.to(self.device), y.to(self.device)
                count += 1
                
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                
                # Накопление статистики для AIC Score
                aic_metric.update(pred, y)
                
        final_aic, dice_pos, fpr_neg = aic_metric.compute()
        
        if self.writer:
            self.writer.add_scalar('Eval/Dice_pos', dice_pos, self.global_step)
            self.writer.add_scalar('Eval/FPR_neg', fpr_neg, self.global_step)
            
        print(f"Валидация завершена -> Val AIC: {final_aic:.4f} | Dice_pos: {dice_pos:.4f} | FPR_neg: {fpr_neg:.4f}")
        
        return val_loss / count, final_aic

    def _log_predictions(self, step):
        if not self.writer:
            return
            
        self.model.eval()
        with torch.no_grad():
            for n, (X, y) in enumerate(self.etalone_samples, 1):
                pred = self.model(X.unsqueeze(0).to(self.device))
                pred = (torch.sigmoid(pred) > 0.8).float()
                
                self.writer.add_image(f'Images/example_{n}/Image', X, step, dataformats='CHW')
                self.writer.add_image(f'Images/example_{n}/Mask', y, step, dataformats='CHW')
                self.writer.add_image(f'Images/example_{n}/Result', pred.cpu().squeeze(0), step, dataformats='CHW')


def main():
    # 1. Настройка окружения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    TOKEN = "y0__wgBEOqqzbcEGIesRyC5wf_SGDDNzNLrB1UKpR0S6zv56ylh9XeIO-G_bAgG"
    
    # 2. Подготовка данных с диска D:
    base_dir = r"D:\train_stage1 (1)\stage1"
    csv_path = os.path.join(base_dir, "train.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Не найден файл разметки по пути: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Собираем абсолютные пути к картинкам и маскам
    data_list = []
    for _, row in df.iterrows():
        img_p = row['chng_img_path']
        mask_p = row['gt_path']
        
        # Умная очистка: если путь в таблице начинается с "stage1/", отрезаем это
        if img_p.startswith('stage1/') or img_p.startswith('stage1\\'):
            img_p = img_p[7:]
        if mask_p.startswith('stage1/') or mask_p.startswith('stage1\\'):
            mask_p = mask_p[7:]
            
        # Теперь безопасно склеиваем с базовой папкой
        if not os.path.isabs(img_p):
            img_p = os.path.join(base_dir, img_p)
        if not os.path.isabs(mask_p):
            mask_p = os.path.join(base_dir, mask_p)
            
        data_list.append((img_p, mask_p))

    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42)

    # 3. Настройка DataLoaders
    train_dataset = TrainPhotosDataset(
        data_list=train_list,
        transforms=[RandomHorizontal(0.25), RandomCrop(0.25), RandomBright(0.25)]
    )
    test_dataset = TrainPhotosDataset(data_list=test_list)

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=False)

    # 4. Настройка облака
    try:
        if TOKEN:
            cloud = CloudManager(token=TOKEN, remote_base_path='/IoU')
        else:
            cloud = None
    except Exception as e:
        print(f"Работаем без облака. Ошибка: {e}")
        cloud = None

    # 5. Эксперимент: Обучение U-Net
    model_unet = UNet(num_classes=1, num_blocks=4)
    optimizer_unet = torch.optim.Adam(model_unet.parameters(), lr=0.00001)
    
    trainer_unet = Trainer(
        model=model_unet,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=CombinedLoss(torch.nn.BCEWithLogitsLoss(), SoftIoULoss(reduction='mean')),
        metric_fn=AICScoreMetric(threshold=0.8),
        optimizer=optimizer_unet,
        device=device,
        log_dir='logs/UNet_AICScore',
        cloud_manager=cloud
    )
    
    t_start = time.time()
    trainer_unet.train(epochs=5)
    print(f"Обучение завершено за {time.time() - t_start:.2f} сек.")


if __name__ == "__main__":
    main()