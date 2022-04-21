import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os




def get_img_dataset(transform=None):
    # Define data transform
    train_transform = []
    if transform is not None:
        train_transform+=transform
    train_transform += [
            transforms.Resize(128),             # resize shortest side to 128 pixels
            transforms.CenterCrop(128),         # crop longest side to 128 pixels at center
            transforms.ToTensor()               # convert PIL image to tensor
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform=transforms.Compose([
            transforms.Resize(128),             # resize shortest side to 128 pixels
            transforms.CenterCrop(128),         # crop longest side to 128 pixels at center
            transforms.ToTensor()               # convert PIL image to tensor
    ])
    
    # Initialize train/test sets
    data_path = Path("data/images")
    train_dataset = ImageFolder(data_path, transform=train_transform)
    test_dataset = ImageFolder(data_path, transform=test_transform)
    classes = train_dataset.find_classes(data_path)[1]
    print(f"Loaded samples into dataset with label 'no'={classes['no']} and 'yes'={classes['yes']}")
    
    # Split dataset into train/test sets and stratify over labels to balance datasets with set seed 
    # DO NOT CHANGE THE SEED
    train_dataset_idx, test_dataset_idx = train_test_split(torch.arange(len(train_dataset)), stratify = train_dataset.targets, test_size = 0.2, random_state=390397)
    train_dataset, test_dataset = Subset(train_dataset, train_dataset_idx), Subset(test_dataset, test_dataset_idx)
    
    return train_dataset, test_dataset
    
    
    
    
def get_radiomics_dataset():
    # Load train/test sets from csvs
    data_path = "data/radiomics"
    train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    train_data.drop(inplace=True,axis=1,labels=['diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'])
    train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    test_data.drop(inplace=True,axis=1,labels=['diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'])
    test_labels = np.load(os.path.join(data_path, 'test_labels.npy'))
    
    return train_data, train_labels, test_data, test_labels