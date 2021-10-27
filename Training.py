
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

########### TRAINING 

######### LOADING MY DATA using the code above 

images = sorted(glob('D:/Dataset/train/swine*.nii'))
segs = sorted(glob('D:/Dataset/train/Label_swine*.nii'))
train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
keys=["img", "seg"]

# define transforms for image and segmentation
train_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        SqueezeDimd(keys=["img", "seg"], dim=-1),
        ScaleIntensityd(keys="img"),
        CenterSpatialCropd(keys=["img", "seg"], roi_size=(300,300)),
        Resized(keys=["img", "seg"], spatial_size=(256,256), mode=('bilinear', 'nearest')),
        RandCropByPosNegLabeld(
             keys=["img", "seg"], label_key="seg", spatial_size=(96, 96), pos=1, neg=1, num_samples=4),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        Rand2DElasticd(keys, mode=('bilinear', 'nearest'), spacing=(10,10), rotate_range=(0.15,0.15), scale_range=(0.05,0.05),
                       magnitude_range=(0,1), prob=0.5),
        
        EnsureTyped(keys=["img", "seg"]),
        ToTensord(keys=["img", "seg"]),
    ]
)


# define dataset, data loader
check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate)
check_data = monai.utils.misc.first(check_loader)
print(check_data["img"].shape, check_data["seg"].shape)

images_val = sorted(glob('D:/Dataset/val/swine*.nii'))
segs_val = sorted(glob('D:/Dataset/val/Label_swine*.nii'))
val_files = [{"img": img, "seg": seg} for img, seg in zip(images_val, segs_val)]

val_transforms = Compose(
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
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
hd_metric   = HausdorffDistanceMetric(include_background=False)
asd_metric  = SurfaceDistanceMetric(include_background=False)

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True), KeepLargestConnectedComponent(applied_labels=[1]), FillHoles()])


# create a training data loader
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
    pin_memory=torch.cuda.is_available(),
)

# create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.UNet(
    dimensions=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)



start_time = time.time()


# start a typical PyTorch training
val_interval = 10
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()
numEpochs = 500
for epoch in range(numEpochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{numEpochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = (96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                hd_metric(y_pred=val_outputs, y=val_labels)
                asd_metric(y_pred=val_outputs, y=val_labels)
                
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_segmentation2d_dict.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                ))
            print("current mean HD: {:.4f}  and current ASD: {:.4f}".format(
                hd_metric.aggregate().item(),asd_metric.aggregate().item()))
            
            writer.add_scalar("val_mean_dice", metric, epoch + 1)

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


print("****it took this much time (in hours):", (time.time()-start_time)/(60*60))

#torch.cuda.empty_cache()



