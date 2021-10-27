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

### TEST PHANTOMDATA WITHOUT ANY LABELS 

######## TESTING ############

model.load_state_dict(torch.load("best_metric_model_segmentation2d_dict.pth"))
images_test = sorted(glob('D:/Dataset/imgs/imgs/*.nii'))
test_files = [{"img": img} for img in zip(images_test)]

test_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        AddChanneld(keys=["img"]),
        SqueezeDimd(keys=["img"], dim=-1),
        ScaleIntensityd(keys="img"),
        CenterSpatialCropd(keys=["img"], roi_size=(200,200)),
        Resized(keys=["img"], spatial_size=(256,256), mode=('bilinear')),
        EnsureTyped(keys=["img"]),
        ToTensord(keys=["img"]),
    ]
)
# create a validation data loader
test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True),
                      KeepLargestConnectedComponent(applied_labels=[1]), FillHoles()])

start_time = time.time()

model.eval()
with torch.no_grad():
    test_images = None
    test_outputs = None
    for i, test_data in enumerate(test_loader):
        test_images = test_data["img"].to(device)
        roi_size = (96, 96)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        
        save_image(test_outputs, 'D:/Dataset/labels/'+str(i)+'.png')
    #end for
#end with
