import os
import torch

# ==========================================
# ПУТИ И ДИРЕКТОРИИ
# ==========================================
BASE_DIR = r"D:\train_stage1 (1)\stage1"
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
LOG_DIR = "logs/UNet_AICScore"

# Токен облака (если используется)
YADISK_TOKEN = "y0__wgBEOqqzbcEGIesRyC5wf_SGDDNzNLrB1UKpR0S6zv56ylh9XeIO-G_bAgG" 

# ==========================================
# ГИПЕРПАРАМЕТРЫ ДАННЫХ
# ==========================================
IMAGE_SIZE = [512, 512]
BATCH_SIZE = 8
NUM_WORKERS = 0
TEST_SPLIT = 0.2
AUGMENT_PROB = 0.25

# ==========================================
# ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ
# ==========================================
EPOCHS = 5
LEARNING_RATE = 1e-5
THRESHOLD = 0.8 # Порог для метрики AIC Score

# ==========================================
# НАСТРОЙКИ АРХИТЕКТУРЫ
# ==========================================
# Варианты: "CustomUNet" (ваша сборка на VGG13) или "SMP" (быстрые сети)
MODEL_NAME = "SMP" 

# Если выбрали "SMP", какой энкодер использовать? (варианты: "mit_b0", "efficientnet-b0")
SMP_ENCODER = "mit_b0" 

# Общие параметры
NUM_CLASSES = 1
NUM_BLOCKS = 4 # Используется только для CustomUNet

# ==========================================
# СИСТЕМНЫЕ НАСТРОЙКИ
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')