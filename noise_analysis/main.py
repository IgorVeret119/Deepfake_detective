from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from train import get_data_list

from .noise_augs import (
    FourierFilter,
    GaussianFilter,
    MedianFilter,
    NoiseExtractor,
    dataset_noise,
    noise_per_pixel,
)
from .vis import plot_img_list, plot_img_noise


def test_filter(
    img_path,
    gt_path,
    filters: list,
    color_basis: Literal["rgb", "ycbcr"] = "rgb",
    inverse_transform=True,
):
    img = Image.open(img_path)
    mask = Image.open(gt_path)
    mask = np.array(mask)

    for filter in filters:
        img_smooth = filter.transform(img)
        noise_extractor = NoiseExtractor(
            filter=filter,
            color_basis=color_basis,
            inverse_transform=inverse_transform,
        )
        noise = noise_extractor.transform(img)
        plot_img_noise(
            noise, img_orig=img, img_smoothed=img_smooth, mask=mask, suptitle=None
        )
        plt.show()


if __name__ == "__main__":
    data_list = get_data_list()
    img_path, gt_path = data_list[1]

    # test_filter(img_path, gt_path, filters=[FourierFilter(sigma=200)])

    # filter = FourierFilter(sigma=400)
    # filter = GaussianFilter(sigma=0.5)
    filter = MedianFilter(kernel_size=3)
    noise_extractor = NoiseExtractor(filter=filter, color_basis="ycbcr")

    res = dataset_noise(data_list[:10000], noise_extractor=noise_extractor)
    print("mean mask noise:", res[0])
    print("mean img noise:", res[1])
    print("mask_noise / img_noise:", res[0] / res[1])
