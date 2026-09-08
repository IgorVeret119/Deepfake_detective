from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


def _plot_img(img, ax=None, title=None):
    ax = plt.gca() if ax is None else ax
    if title is not None:
        ax.set_title(title)

    ax.axis("off")
    ax.imshow(img)


def plot_img_list(
    imgs: list,
    nrows=-1,
    ncols=2,
    suptitle=None,
    img_titles: list | None = None,
):
    img_count = len(imgs)

    if nrows == -1 and ncols == -1:
        nrows = int(np.sqrt(img_count))

    nrows = int(np.ceil(img_count / ncols)) if nrows == -1 else nrows
    ncols = int(np.ceil(img_count / nrows)) if ncols == -1 else ncols

    fig, axs = plt.subplots(figsize=(ncols * 3, nrows * 3), nrows=nrows, ncols=ncols)
    axs = np.atleast_1d(axs)

    suptitle = "" if suptitle is None else suptitle
    fig.suptitle(suptitle)

    img_titles = [None] * img_count if img_titles is None else img_titles

    for img, ax, title in zip(imgs, axs.ravel(), img_titles):
        _plot_img(img, ax=ax, title=title)

    for ax in axs.ravel()[img_count:]:
        ax.remove()

    fig.tight_layout()


def _plot_noise(img, ax=None, title=None):
    ax = plt.gca() if ax is None else ax
    if title is not None:
        ax.set_title(title)

    ax.axis("off")
    v = np.percentile(img, 99)
    ax.imshow(img, cmap="RdBu_r", vmin=0, vmax=v)


def plot_mask(mask, ax=None, how: Literal["contour", "fill"] = "contour", color="red"):
    ax = plt.gca() if ax is None else ax
    if how == "contour":
        ax.contour(mask, levels=[0.5], linewidths=1, colors=color)
    else:
        ax.contourf(mask, levels=[0.5, 255], colors=color)


def plot_img_noise(
    img_noise,
    img_orig=None,
    img_smoothed=None,
    mask=None,
    suptitle: str | None = "Image noise visualisation",
):
    ax_count = 1
    if img_orig is not None:
        ax_count += 1
    if img_smoothed is not None:
        ax_count += 1

    fig, axs = plt.subplots(figsize=(6 * ax_count, 8), ncols=ax_count)
    if suptitle is not None:
        fig.suptitle(suptitle)
    axs = np.atleast_1d(axs)

    _plot_noise(img_noise, ax=axs[0], title="Image Noise")

    c_ax = 1  # current ax
    if img_orig is not None:
        axs[c_ax].imshow(img_orig)
        axs[c_ax].set_title("Original Image")
        axs[c_ax].axis("off")
        c_ax += 1

    if img_smoothed is not None:
        axs[c_ax].imshow(img_smoothed)
        axs[c_ax].set_title("Smoothed Image")
        axs[c_ax].axis("off")
        c_ax += 1

    if mask is not None:
        if len(mask.shape) == 3:
            mask = mask.mean(axis=-1, dtype=int)
        for ax in axs:
            plot_mask(mask, ax=ax, how="contour", color="yellow")

        ground_truth = Line2D(
            [0], [0], color="yellow", linewidth=1, label="Ground Truth"
        )
        fig.legend(handles=[ground_truth])

    fig.tight_layout(pad=2)
