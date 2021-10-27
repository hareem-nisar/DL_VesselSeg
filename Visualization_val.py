from pathlib import Path

import logging
import os
import sys
import tempfile
from glob import glob
import shutil
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric,SurfaceDistanceMetric
from monai.transforms import (
    KeepLargestConnectedComponent,
    FillHoles,
    ToTensord,
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandAffined,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SqueezeDimd,
    Resized,
    CropForegroundd,
    Rand2DElasticd,
)
from monai.visualize import plot_2d_or_3d_image


import nibabel as nib
import numpy as np
from monai.config import print_config
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.utils import first

import ignite
import torch
from torchvision.utils import save_image

import matplotlib.pyplot as plt


# For visualozation only 

model.load_state_dict(torch.load("best_metric_model_segmentation2d_dict.pth"))
model.eval()

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (96, 96)
        sw_batch_size = 4
        val_label = val_data["seg"].to(device)
        val_data = val_data["img"].to(device)
        val_output = sliding_window_inference(
            val_data, roi_size, sw_batch_size, model)
        # plot the slice [:, :, 80]
        plt.figure("check", (20, 4))
        plt.subplot(1, 5, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data.detach().cpu()[0, 0, :, :], cmap="gray")
        plt.subplot(1, 5, 2)
        plt.title(f"argmax {i}")
        argmax = [AsDiscrete(threshold_values=True)(i) for i in decollate_batch(val_output)]
        plt.imshow(argmax[0].detach().cpu()[0, :, :], cmap="Set3")
        plt.subplot(1, 5, 3)
        plt.title(f"largest {i}")
        largest = [KeepLargestConnectedComponent(applied_labels=[1])(i) for i in argmax]
        plt.imshow(largest[0].detach().cpu()[0, :, :], cmap="Set3")
        plt.subplot(1, 5, 4)
        plt.title(f"Filled Holes {i}")
        filled = [FillHoles()(i) for i in largest]
        plt.imshow(filled[0].detach().cpu()[0, :, :], cmap="Set3")
        plt.subplot(1, 5, 5)
        plt.title(f"map image {i}")
        map_image = val_label[0] + filled[0] #+ val_data[0]  
        #plt.imshow(map_image.detach().cpu()[0, :, :], cmap = "Pastel1")
        #plt.imshow(val_data[0].detach().cpu()[0,:,:], alpha = 0.3, cmap = "gray")
        plt.imshow(val_label[0].detach().cpu()[0,:,:], alpha = 0.7, cmap = "GnBu")
        plt.imshow(filled[0].detach().cpu()[0,:,:], alpha = 0.7, cmap = "Set3")
        plt.savefig(f"UNET{i}.png")
        plt.show()