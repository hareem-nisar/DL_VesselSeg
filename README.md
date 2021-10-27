# DL_VesselSeg
A MONAI-framework based vessel segmentation algorithm compatible with ICE imaging of inferior vena cava (IVC)
The following files can be used with Jupyter notebook and another python platform provided that MONAI and its dependencies are installed. For installation of MONAI refer to this link https://docs.monai.io/en/latest/installation.html

This work was done with MONAI v0.6 and is compatible with the stable version v0.7. 

**Training.py** : Loads training dataset containing images and labels, pre-processes images, performs data augmentation, train a U-net, post-process the output label, test the network accuracy every few iterations using the validation dataset. (Note: Training and validation dataset not provided.) 

**Visulaization_val.py** : Runs the validation dataset through the pre-trained U-net model and shows the output at different post-processing stages. Shows the comparison of the network output overlayed with the ground truth label. (Note: Pre-trained model is provided.)

**Testing_withLabels.py** : Runs the training dataset through the pre-trained U-net model and saves the output labels locally. Computes and saves Dice coefficient. (Note: Pre-trained model parameters and test dataset is provided.)

**Visulaization_test.py** : Runs the training dataset through the pre-trained U-net model and shows the output at different post-processing stages. Shows the comparison of the network output overlayed with the ground truth label. (Note: Pre-trained model and testing dataset is provided.)

**Testing_withoutLabels.py** : Runs your new image-only dataset through the pre-trained U-net model and saves the output labels locally in .png format.

**best_metric_model_segmentation2d_dict.pth** : Pre-trained U-net model parameters with 96% accuracy.
