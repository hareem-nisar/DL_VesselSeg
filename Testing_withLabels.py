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

######## TESTING ############

model.load_state_dict(torch.load("runs/13. Final Model 96p/best_metric_model_segmentation2d_dict.pth"))
metric_values = list()
images_test = sorted(glob('D:/Dataset/test/swine*.nii'))
segs_test = sorted(glob('D:/Dataset/test/Label_swine*.nii'))
test_files = [{"img": img, "seg": seg} for img, seg in zip(images_test, segs_test)]

test_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        SqueezeDimd(keys=["img", "seg"], dim=-1),
        ScaleIntensityd(keys="img"),
        CenterSpatialCropd(keys=["img", "seg"], roi_size=(300,300)),
        Resized(keys=["img", "seg"], spatial_size=(256,256), mode=('bilinear', 'nearest')),
        EnsureTyped(keys=["img", "seg"]),
        ToTensord(keys=["img", "seg"]),
    ]
)
# create a validation data loader
test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
hd_metric   = HausdorffDistanceMetric(include_background=False)
asd_metric  = SurfaceDistanceMetric(include_background=False)

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True),
                      KeepLargestConnectedComponent(applied_labels=[1]), FillHoles()])

start_time = time.time()

model.eval()
with torch.no_grad():
    test_images = None
    test_labels = None
    test_outputs = None
    for i, test_data in enumerate(test_loader):
        test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(device)
        roi_size = (96, 96)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        # compute metric for current iteration
        dice_metric(y_pred=test_outputs, y=test_labels)
        hd_metric(  y_pred=test_outputs, y=test_labels)
        asd_metric( y_pred=test_outputs, y=test_labels)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        metric_values.append([dice_metric.aggregate().item(),hd_metric.aggregate().item(),asd_metric.aggregate().item() ])
        print(" current mean dice: {:.4f}, current mean HD: {:.4f}  and current ASD: {:.4f}".format(
        dice_metric.aggregate().item(), hd_metric.aggregate().item(), asd_metric.aggregate().item()))
        
        # SAVE LMs to a local file
        #nib.save(nib.Nifti1Image(test_outputs[0][0].cpu().numpy(), affine=np.eye(4)), "out"+str(i)+".nii")
        
    
    dice_metric.reset()
    hd_metric.reset()
    asd_metric.reset()
    print()
    print(metric_values)

x= np.asarray(metric_values)
np.savetxt("test_results.csv", x, delimiter=",")


