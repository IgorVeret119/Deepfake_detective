from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class BaseFilter(BaseEstimator, TransformerMixin):
    pass


import numpy as np
from scipy.ndimage import median_filter


class MedianFilter(BaseFilter):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("Wrong kernel size!")

        return self

    def transform(self, X):
        X = np.asarray(X)

        if X.ndim not in (2, 3):
            raise ValueError("Wrong image dimension count!")

        if X.ndim == 2:
            size = self.kernel_size
        else:
            # Don't touch RGB
            size = (self.kernel_size, self.kernel_size, 1)

        return median_filter(
            X,
            size=size,
            mode="reflect",
        )


class GaussianFilter(BaseFilter):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def fit(self, X, y=None):
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive!")
        return self

    def transform(self, X):
        X = np.asarray(X)

        if X.ndim not in (2, 3):
            raise ValueError("Wrong image dimension count!")

        if X.ndim == 2:
            return gaussian_filter(X, sigma=self.sigma)

        return gaussian_filter(X, sigma=(self.sigma, self.sigma, 0))


class FourierFilter(BaseFilter):
    def __init__(self, sigma=10.0):
        self.sigma = sigma

    def fit(self, X, y=None):
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive!")
        return self

    def transform(self, X):
        X = np.asarray(X)

        if X.ndim not in (2, 3):
            raise ValueError("Wrong image dimension count!")

        if X.ndim == 2:
            return self._filter_channel(X)

        channels = [self._filter_channel(X[:, :, c]) for c in range(X.shape[2])]
        X_filtered = np.stack(channels, axis=2)
        X_filtered = np.asarray(np.clip(X_filtered, a_min=0, a_max=255), dtype=np.uint8)
        return X_filtered

    def _filter_channel(self, channel):
        channel = channel.astype(np.float32)

        h, w = channel.shape
        F = np.fft.fft2(channel)
        F = np.fft.fftshift(F)

        # Frequency coordinates
        y = np.arange(-(h // 2), h - (h // 2))
        x = np.arange(-(w // 2), w - (w // 2))
        y, x = np.meshgrid(y, x, indexing="ij")

        # Distance from zero frequency
        D2 = x**2 + y**2

        # Gaussian low-pass filter
        H = np.exp(-D2 / (2 * self.sigma**2))

        # Filtering in frequency domain
        F_filtered = F * H

        # Inverse Fourier transform
        F_filtered = np.fft.ifftshift(F_filtered)
        result = np.fft.ifft2(F_filtered)

        return np.real(result)


class NoiseExtractor(BaseFilter):
    def __init__(
        self,
        filter=None,
        color_basis: Literal["rgb", "ycbcr"] = "rgb",  # apply filter in this basis
        inverse_transform=True,  # return to the original database after applying the filter
    ):
        if filter is None:
            filter = MedianFilter()
        self.filter = filter
        self.color_basis = color_basis
        self.inverse_transform = inverse_transform

    def transform(self, X):
        X = np.asarray(X)

        if self.color_basis == "ycbcr":
            X = cv2.cvtColor(X, cv2.COLOR_RGB2YCrCb)

        X_filtered = self.filter.transform(X)

        if self.color_basis == "ycbcr" and self.inverse_transform:
            X = cv2.cvtColor(X, cv2.COLOR_YCrCb2RGB)
            X_filtered = cv2.cvtColor(X_filtered, cv2.COLOR_YCrCb2RGB)

        # prevent overflow
        X_diff = np.asarray(X, dtype=np.float32) - np.asarray(
            X_filtered, dtype=np.float32
        )

        if len(X.shape) == 3:
            noise = np.linalg.norm(X_diff, axis=-1)
        else:
            noise = np.abs(X_diff)
        return noise


def noise_per_pixel(img_noise, mask=None):
    img_noise = np.asarray(img_noise)

    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        mask = mask > 0

        area = mask.sum()
        if area == 0:
            return 0.0
        return img_noise[mask].sum() / area

    area = img_noise.shape[0] * img_noise.shape[1]
    return img_noise.sum() / area


def _process_one_noise_calc(args):
    img_path, gt_path, noise_extractor = args
    img = Image.open(img_path)
    mask = Image.open(gt_path)

    if img.size[:2] != mask.size[:2]:
        print(
            f"Warning: different image and mask sizes: {img.size} vs {mask.size}. Skipped."
        )
        return None

    noise = noise_extractor.transform(img)
    mask_noise = noise_per_pixel(noise, mask=mask)
    img_noise = noise_per_pixel(noise)
    return mask_noise, img_noise


def dataset_noise(data_list: list[tuple], noise_extractor=None, max_workers=None):
    if noise_extractor is None:
        noise_extractor = NoiseExtractor()

    mask_noise_sum = 0.0
    img_noise_sum = 0.0
    processed = 0

    tasks = [(img_path, gt_path, noise_extractor) for img_path, gt_path in data_list]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one_noise_calc, task) for task in tasks]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Noise calculation"
        ):
            result = future.result()
            if result is not None:
                mask_noise, img_noise = result
                mask_noise_sum += mask_noise
                img_noise_sum += img_noise
                processed += 1

    if processed == 0:
        return (0, 0)

    return mask_noise_sum / processed, img_noise_sum / processed
