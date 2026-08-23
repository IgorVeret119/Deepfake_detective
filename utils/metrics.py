import torch
from torch import nn

class DiceLoss(nn.Module):
    """
    Функция потерь Dice Loss. Дифференцируема, используется для обучения.
    """
    def __init__(self, eps=1e-7, reduction=None, with_logits=True):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.with_logits = with_logits

    def forward(self, logits, true_labels):
        true_labels = true_labels.to(logits.dtype)

        if self.with_logits:
            logits = torch.sigmoid(logits)
            
        axis = tuple(range(1, logits.ndim))
        
        intersection = (2 * logits * true_labels).sum(dim=axis)
        pred_sum = logits.sum(dim=axis)
        true_sum = true_labels.sum(dim=axis)
        
        res = 1.0 - intersection / (pred_sum + true_sum + self.eps)

        if self.reduction == 'sum':
            return res.sum()
        elif self.reduction == 'mean':
            return res.mean()
        
        return res


class IoUScore(nn.Module):
    """
    Метрика IoU (Индекс Жаккара). 
    ВНИМАНИЕ: Не имеет градиентов! Использовать только для оценки (валидации), а не для обучения.
    """
    def __init__(self, threshold=0.5, reduction=None, with_logits=True, eps=1e-7):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.with_logits = with_logits
        self.eps = eps

    @torch.no_grad()
    def forward(self, logits, true_labels):
        if self.with_logits:
            logits = torch.sigmoid(logits)
            
        # Бинаризация (превращает вероятности в жесткие 0 и 1)
        logits = (logits > self.threshold).float()
        true_labels = true_labels.float()

        axis = tuple(range(1, logits.ndim))

        intersection = (logits * true_labels).sum(dim=axis)
        pred_sum = logits.sum(dim=axis)
        true_sum = true_labels.sum(dim=axis)
        
        union = pred_sum + true_sum - intersection
        score = intersection / (union + self.eps)

        if self.reduction == 'sum':
            return torch.nansum(score)
        elif self.reduction == 'mean':
            return torch.nanmean(score)
            
        return score


class SoftIoULoss(nn.Module):
    """
    Дифференцируемая версия IoU (Jaccard Loss). 
    Можно использовать как функцию потерь (Loss) для обучения нейросети.
    В отличие от IoUScore, здесь нет жесткого порога (threshold).
    """
    def __init__(self, eps=1e-7, reduction='mean', with_logits=True):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.with_logits = with_logits

    def forward(self, logits, true_labels):
        true_labels = true_labels.to(logits.dtype)

        if self.with_logits:
            logits = torch.sigmoid(logits)
            
        axis = tuple(range(1, logits.ndim))

        # Используем сами вероятности (без > threshold)
        intersection = (logits * true_labels).sum(dim=axis)
        pred_sum = logits.sum(dim=axis)
        true_sum = true_labels.sum(dim=axis)
        
        union = pred_sum + true_sum - intersection
        
        # Loss = 1 - SoftIoU
        # Добавляем eps и в числитель, и в знаменатель для стабильности
        res = 1.0 - (intersection + self.eps) / (union + self.eps)

        if self.reduction == 'sum':
            return res.sum()
        elif self.reduction == 'mean':
            return res.mean()
            
        return res


class CombinedLoss(nn.Module):
    """
    Класс для объединения нескольких функций потерь с заданными весами.
    Пример использования: 
    loss = CombinedLoss(BCEWithLogitsLoss(), SoftIoULoss(), weight1=1.0, weight2=2.0)
    """
    def __init__(self, loss1, loss2, weight1=1.0, weight2=1.0):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, pred, target):
        return (self.weight1 * self.loss1(pred, target)) + (self.weight2 * self.loss2(pred, target))


class AICScoreMetric:
    """
    Класс для расчета кастомной метрики AIC (Harmonic mean of Dice_pos and 1 - FPR_neg).
    Используется для оценки на этапе валидации (evaluation).
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Сброс счетчиков в начале каждой эпохи валидации."""
        self.dice_sum = 0.0
        self.pos_count = 0
        
        self.fp_count = 0.0
        self.neg_count = 0

    @torch.no_grad()
    def update(self, logits, true_labels):
        """
        Метод вызывается на каждом батче. Накапливает статистику.
        """
        # Переводим сырые логиты в бинарную маску (0 или 1)
        preds = (torch.sigmoid(logits) > self.threshold).float()
        true_labels = true_labels.float()
        
        # Оси для суммирования (все пространственные размерности: C, H, W)
        axis = tuple(range(1, preds.ndim))
        
        # Считаем сумму пикселей в Ground Truth, чтобы разделить батч на Pos и Neg
        gt_sum = true_labels.sum(dim=axis)
        
        # Маски для разделения (True/False)
        is_pos = gt_sum > 0  # Есть изменения (GT не пустой)
        is_neg = gt_sum == 0 # Нет изменений (GT пустой)
        
        # =========================================================
        # 1. Расчет Dice для Positive-примеров
        # =========================================================
        if is_pos.any():
            pos_preds = preds[is_pos]
            pos_gts = true_labels[is_pos]
            
            intersection = (pos_preds * pos_gts).sum(dim=axis)
            p_sum = pos_preds.sum(dim=axis)
            g_sum = pos_gts.sum(dim=axis)
            
            # Формула из скриншота: (2 * |P_i \cap G_i|) / (|P_i| + |G_i| + 10^-6)
            dice_i = (2.0 * intersection) / (p_sum + g_sum + 1e-6)
            
            self.dice_sum += dice_i.sum().item()
            self.pos_count += is_pos.sum().item()
            
        # =========================================================
        # 2. Расчет FPR для Negative-примеров
        # =========================================================
        if is_neg.any():
            neg_preds = preds[is_neg]
            
            # Общее количество пикселей в одном кадре (H * W)
            total_pixels = neg_preds[0].numel() 
            
            # Площадь предсказанной маски
            p_sum = neg_preds.sum(dim=axis)
            area = p_sum / total_pixels
            
            # Индикатор: 1, если площадь >= 0.01 (1% кадра), иначе 0
            false_alarms = (area >= 0.01).float()
            
            self.fp_count += false_alarms.sum().item()
            self.neg_count += is_neg.sum().item()

    def compute(self):
        """
        Вызывается в конце эпохи для расчета итогового AIC Score.
        """
        # Усредняем Dice для позитивных примеров
        dice_pos = self.dice_sum / self.pos_count if self.pos_count > 0 else 0.0
        
        # Усредняем FPR для негативных примеров
        fpr_neg = self.fp_count / self.neg_count if self.neg_count > 0 else 0.0
        
        # =========================================================
        # 3. Итоговый AIC Score (Гармоническое среднее)
        # =========================================================
        component2 = 1.0 - fpr_neg
        
        # Защита от деления на ноль, если обе метрики провалены (равны нулю)
        if dice_pos + component2 == 0:
            aic = 0.0
        else:
            # Формула: (2 * Dice_pos * (1 - FPR_neg)) / (Dice_pos + (1 - FPR_neg))
            aic = (2.0 * dice_pos * component2) / (dice_pos + component2)
            
        return aic, dice_pos, fpr_neg
