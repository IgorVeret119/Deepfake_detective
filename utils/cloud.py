import os
import torch
import yadisk
import tempfile
import config

class CloudManager:
    """
    Класс для управления загрузкой и сохранением чекпоинтов и логов на Яндекс Диск.
    """
    def __init__(self, token, remote_base_path='/IoU'):
        """
        Аргументы:
            token (str): OAuth-токен от Яндекс Диска.
            remote_base_path (str): Базовая папка на Диске для этого эксперимента.
        """
        self.client = yadisk.YaDisk(token=token)
        self.remote_base_path = remote_base_path
        
        # Проверяем токен и создаем базовую папку, если её нет
        if not self.client.check_token():
            raise ValueError("Неверный токен Яндекс Диска!")
            
        if not self.client.exists(self.remote_base_path):
            self.client.mkdir(self.remote_base_path, recursive=True)
            print(f"Создана базовая директория на Диске: {self.remote_base_path}")

    def load_latest_checkpoint(self, model, optimizer, device, local_path="/tmp/checkpoint_latest.pth"):
        """
        Скачивает последний чекпоинт с Диска и восстанавливает состояние модели и оптимизатора.
        Возвращает номер эпохи, с которой нужно продолжить обучение.
        """
        remote_checkpoint = f"{self.remote_base_path}/checkpoint_latest.pth"
        start_epoch = 0

        if self.client.exists(remote_checkpoint):
            print(f"Загрузка чекпоинта с Яндекс Диска: {remote_checkpoint}")
            self.client.download(remote_checkpoint, local_path)

            checkpoint = torch.load(local_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"Чекпоинт загружен. Продолжаем с эпохи {start_epoch}")
            os.remove(local_path)
        else:
            print("Чекпоинт на Яндекс Диске не найден. Начинаем обучение с нуля.")

        return start_epoch

    def save_checkpoint(self, epoch, model, optimizer, val_loss, val_metric):        
        # 1. Берем локальную папку для чекпоинтов из конфига
        local_dir = config.CHECKPOINT_DIR
        os.makedirs(local_dir, exist_ok=True)
        
        # 2. Формируем путь к файлу (теперь без /tmp, работает и на Windows, и на Linux)
        save_path = os.path.join(local_dir, f"checkpoint_epoch_{epoch}.pth")
        
        # 3. Сохраняем все важные данные (веса, шаг оптимизатора, метрики)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metric': val_metric
        }, save_path)
        
        print(f"[*] Чекпоинт сохранен локально: {save_path}")

        # 4. Загружаем файл в облако (если клиент настроен)
        try:
            remote_path = f"{self.remote_base_path}/checkpoint_epoch_{epoch}.pth"
            
            # Если вы используете библиотеку yadisk, отправка выглядит примерно так:
            if hasattr(self, 'client') and self.client.check_token():
                self.client.upload(save_path, remote_path, overwrite=True)
                print(f"[*] Чекпоинт успешно отправлен в облако: {remote_path}")
            else:
                print("[!] Подключение к облаку отсутствует. Модель сохранена только локально.")
                
        except Exception as e:
            print(f"[!] Ошибка при загрузке в облако: {e}")
            print(f"[*] Не переживайте, веса в безопасности на вашем диске: {save_path}")

    def sync_logs(self, local_log_dir):
        """
        Отправляет все логи TensorBoard из локальной папки на Яндекс Диск.
        """
        remote_logs = f"{self.remote_base_path}/logs"
        
        if not self.client.exists(remote_logs):
            self.client.mkdir(remote_logs, recursive=True)

        print(f"Синхронизация логов с Яндекс Диском...")
        for root, dirs, files in os.walk(local_log_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_file_path, local_log_dir)
                remote_file_path = f"{remote_logs}/{rel_path}"

                # Создаем подпапки на диске, если нужно
                remote_dir = os.path.dirname(remote_file_path)
                if not self.client.exists(remote_dir):
                    self.client.mkdir(remote_dir, recursive=True)

                self.client.upload(local_file_path, remote_file_path, overwrite=True)
        print(f"Логи успешно синхронизированы.")