import os, time

from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from utils import *

class MOODDataset(Dataset):
    """MICCAI 2021 MOOD Challange Dataset""" 
    def __init__(self,
                 dataset_path,
                 subset='train',
                 category = 'brain'):

        self.idx = 1

        self.dataset_path = os.path.join(dataset_path, subset)
        self.subset = subset
        self.category = category

        self.img_names = [f for f in os.listdir(self.dataset_path) if f.endswith('.nii.gz')]
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = self.GetImage(self.img_names[idx])
                   
        if self.category == 'brain':
            img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls  = extract_brain_patch(img,                                                
                                                            self.idx,
                                                            self.subset)
        elif self.category == 'abdom':
            img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls = extract_abdom_patch(img, 
                                                            self.idx,
                                                            self.subset)
        
        img_patch[img_patch<0.0] = 0.0
        img_patch[img_patch>1.0] = 1.0
        self.idx += 1
        aug_patch[aug_patch<0.0] = 0.0
        aug_patch[aug_patch>1.0] = 1.0
        return img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls
     
    def GetImage(self, image_name):
        image_path = os.path.join(self.dataset_path, image_name)
        img  = nib.load(image_path)
        img_data = img.get_fdata()
        img_data = np.array(img_data, dtype = np.float32)
        return img_data
