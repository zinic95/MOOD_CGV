import nibabel as nib
import random
import numpy as np
import torch
from scipy.ndimage import zoom
from augmentations import *

# Training
def extract_brain_patch(img, i, mode):
    patch_size = 64
    while True:
        # crop the patch
        # normal
        img_patch = np.copy(img)
        imgGT_patch = np.zeros_like(img_patch)
        aug_cls = 1.0
        # anomaly (augmented)
        aug_patch = np.copy(img_patch)
        augGT_patch = np.zeros_like(img_patch)

        if mode == 'train':
            if np.random.rand() > 0.5:
                img_patch, aug_patch, = elastic_transform(img_patch, aug_patch, alpha=20, sigma=5)
            #Data Augmentation            
            if i % 2 == 0:
                aug_patch, augGT_patch = copy_n_paste(aug_patch, augGT_patch, patch_size, patch_size/2, fg_th = 0.3)
            elif i % 2 == 1:
                aug_patch, augGT_patch = copy_n_paste_rot(aug_patch, augGT_patch, patch_size, patch_size/2, fg_th = 0.3)

            #Color Jittering
            if  np.random.rand() > 0.5:
                aug_patch[augGT_patch == 1] += (np.random.rand() - 0.5) * 2 / 5 # -0.2~0.2 
                aug_patch[aug_patch<0.0] = 0.0
                aug_patch[aug_patch>1.0] = 1.0
          
            augGT_patch  = removeBGfromGT(img_patch, aug_patch, augGT_patch) 
            if ( np.count_nonzero(augGT_patch) < 20 ) :
                aug_cls = 0.0
            else :
                aug_cls = 1.0
        elif mode == 'valid':
            #Data Augmentation
            if i % 6 == 1:
                aug_patch, augGT_patch = random_masking(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 2: 
                aug_patch, augGT_patch = sobel_filter(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 3:
                aug_patch, augGT_patch = rotate(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 4:
                aug_patch, augGT_patch = copy_n_paste(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 5:
                aug_patch, augGT_patch = permutation(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 0:
                aug_patch, augGT_patch = copy_n_paste_scar(aug_patch, augGT_patch, patch_size, patch_size/4)
            augGT_patch  = removeBGfromGT(img_patch, aug_patch, augGT_patch)
            if ( np.count_nonzero(augGT_patch) < 20 ) :
                aug_cls = 0.0
            else :
                aug_cls = 1.0
                
        if aug_cls == 1.0:
            break

    img_patch = np.expand_dims(img_patch, axis=0)
    imgGT_patch = np.expand_dims(imgGT_patch, axis=0)
    aug_patch = np.expand_dims(aug_patch, axis=0)
    augGT_patch = np.expand_dims(augGT_patch, axis=0)
    return img_patch, imgGT_patch, 0.0, aug_patch, augGT_patch, aug_cls


def extract_abdom_patch(img, i, mode):
    patch_size = 128
    while True:
        stride = int(patch_size/2)
        position = random.randint(0, 26)
        x = position // 9
        y = (position - 9*x) // 3
        z = (position - 9*x - 3*y) % 3
        x = stride * x
        y = stride * y
        z = stride * z
        
        # crop the patch
        # normal
        img_patch = img[x:x + patch_size, y:y + patch_size, z:z + patch_size]
        imgGT_patch = np.zeros_like(img_patch)
        aug_cls = 1.0
        # anomaly (augmented)
        aug_patch = np.copy(img_patch)
        augGT_patch = np.zeros_like(img_patch)
        
        if (np.count_nonzero(img_patch) > 125):
            break
            
    while True:     
        if mode == 'train':
            if np.random.rand() > 0.5:
                img_patch, aug_patch, = elastic_transform(img_patch, aug_patch, alpha=20, sigma=5)
            #Data Augmentation            
            if i % 2 == 0:
                aug_patch, augGT_patch = copy_n_paste(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 2 == 1:
                aug_patch, augGT_patch = copy_n_paste_rot(aug_patch, augGT_patch, patch_size, patch_size/2)
                
            if  np.random.rand() > 0.5:
                aug_patch[augGT_patch == 1] += (np.random.rand() - 0.5) * 2 / 5 # -0.2~0.2 
                aug_patch[aug_patch<0.0] = 0.0
                aug_patch[aug_patch>1.0] = 1.0
        
            augGT_patch  = removeBGfromGT(img_patch, aug_patch, augGT_patch) 
            if ( np.count_nonzero(augGT_patch) < 20 ) :
                aug_cls = 0.0
            else :
                aug_cls = 1.0
        elif mode == 'valid':
            #Data Augmentation
            if i % 6 == 1:
                aug_patch, augGT_patch = random_masking(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 2: 
                aug_patch, augGT_patch = sobel_filter(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 3:
                aug_patch, augGT_patch = rotate(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 4:
                aug_patch, augGT_patch = copy_n_paste(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 5:
                aug_patch, augGT_patch = permutation(aug_patch, augGT_patch, patch_size, patch_size/2)
            elif i % 6 == 0:
                aug_patch, augGT_patch = copy_n_paste_scar(aug_patch, augGT_patch, patch_size, patch_size/4)
           
            augGT_patch  = removeBGfromGT(img_patch, aug_patch, augGT_patch)
            if ( np.count_nonzero(augGT_patch) < 20 ) :
                aug_cls = 0.0
            else :
                aug_cls = 1.0

        if aug_cls == 1.0 :
            break
    
    img_patch = np.expand_dims(img_patch, axis=0)
    imgGT_patch = np.expand_dims(imgGT_patch, axis=0)
    aug_patch = np.expand_dims(aug_patch, axis=0)
    augGT_patch = np.expand_dims(augGT_patch, axis=0)
    
    label = np.zeros((27,patch_size,patch_size,patch_size))
    label[position,:,:,:] = 1
    img_patch = np.concatenate((img_patch, label), axis=0)
    aug_patch = np.concatenate((aug_patch, label), axis=0)
    return img_patch, imgGT_patch, 0.0, aug_patch, augGT_patch, aug_cls

def Segloss(pred, target, patch_size):
    FP = torch.sum((1 - target) * pred) / (patch_size*patch_size*patch_size)
    FN = torch.sum((1 - pred) * target) / (patch_size*patch_size*patch_size)
    return FP, FN

# Testing
def resize_brain(img, target_size):
    affine = nib.aff2axcodes(img.affine) 
    orientation = nib.orientations.axcodes2ornt(affine)
    orientation_rev = np.zeros((3,2))
    for i in range(3):
        idx = np.where(orientation[:,0] == i)[0][0]
        orientation_rev[i,:] = np.array([idx, orientation[idx,1]]) 
    canonical_img = nib.as_closest_canonical(img)
    img_data = canonical_img.get_fdata()

    target_shape = [target_size / img_data.shape[0], target_size / img_data.shape[1], target_size / img_data.shape[2]]

    resized_img = zoom(img_data, target_shape, order = 0)
    resized_img[resized_img < 0] = 0
    resized_img[resized_img > 1] = 1
    
    return resized_img, orientation_rev

def resize_abdom(img, target_size):
    affine = nib.aff2axcodes(img.affine) 
    orientation = nib.orientations.axcodes2ornt(affine)
    orientation_rev = np.zeros((3,2))
    for i in range(3):
        idx = np.where(orientation[:,0] == i)[0][0]
        orientation_rev[i,:] = np.array([idx, orientation[idx,1]]) 
    canonical_img = nib.as_closest_canonical(img)
    img_data = canonical_img.get_fdata()
    target_shape = [target_size / img_data.shape[0], target_size / img_data.shape[1], target_size / img_data.shape[2]]

    resized_img = zoom(img_data, target_shape, order = 0)
    
    resized_img[resized_img < 0] = 0
    resized_img[resized_img > 1] = 1

    return resized_img, orientation_rev

def recon(img, orig_shape, orientation_rev):
    target_shape = [orig_shape[0] / img.shape[0], orig_shape[1] / img.shape[1], orig_shape[2] / img.shape[2]]
    recon_img = nib.orientations.apply_orientation(img, orientation_rev)
    resized_img = zoom(recon_img, target_shape, order = 1)
    resized_img[resized_img < 0] = 0
    resized_img[resized_img > 1] = 1

    return resized_img