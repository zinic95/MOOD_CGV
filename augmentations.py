import random
from re import X
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

# Hard Augmentaion

def rotate(image, gt, patch_size, mask_max):
    rot1, rot2, rot3 = 0, 0, 0
    while rot1 + rot2 + rot3 == 0 or (rot1==2 and rot2==2 and rot3==2):
        rot1 = random.randrange(4)
        rot2 = random.randrange(4)
        rot3 = random.randrange(4)
    mask_size = random.randint(10, mask_max)
    mask_pos = [random.randint(0, patch_size - mask_size) for i in range(len(image.shape))]
    rotated_mask = np.zeros((mask_size, mask_size, mask_size))
    rotated_mask = image[mask_pos[0]:mask_pos[0] + mask_size, mask_pos[1]:mask_pos[1] + mask_size, mask_pos[2]:mask_pos[2] + mask_size]
    rotated_mask = np.rot90(rotated_mask, rot1, (0,1))
    rotated_mask = np.rot90(rotated_mask, rot2, (1,2))
    rotated_mask = np.rot90(rotated_mask, rot3, (2,0))
    
    image[mask_pos[0]:mask_pos[0] + mask_size, mask_pos[1]:mask_pos[1] + mask_size, mask_pos[2]:mask_pos[2] + mask_size] = rotated_mask
    gt[mask_pos[0]:mask_pos[0] + mask_size, mask_pos[1]:mask_pos[1] + mask_size, mask_pos[2]:mask_pos[2] + mask_size] = 1

    return image, gt

def random_masking(image, gt, patch_size, mask_max):
    intensity = random.uniform(0,1)
    mask_size = [random.randint(10, mask_max) for _ in range(len(image.shape))]
    mask_pos = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
    image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]] = intensity
    gt[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]] = 1

    return image, gt

# Do not overlap the copied mask at least half of mask_size for one dimension
def copy_n_paste(image, gt, patch_size, mask_max, fg_th = 0.0):
    while True:
        mask_size = [random.randint(10, mask_max) for _ in range(len(image.shape))]
        mask_pos = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
        mask_pos2 = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
        overlap = [abs(mask_pos[i] -mask_pos2[i]) < mask_size[i]/2 for i in range(len(image.shape))]
        if overlap == [True] * len(image.shape):
            continue
        if np.count_nonzero(image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]]) / (mask_size[0]*mask_size[1]*mask_size[2]) < fg_th:
            continue
        image[mask_pos2[0]:mask_pos2[0] + mask_size[0], mask_pos2[1]:mask_pos2[1] + mask_size[1], mask_pos2[2]:mask_pos2[2] + mask_size[2]] = image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]]
        gt[mask_pos2[0]:mask_pos2[0] + mask_size[0], mask_pos2[1]:mask_pos2[1] + mask_size[1], mask_pos2[2]:mask_pos2[2] + mask_size[2]] = 1
        return image, gt

def copy_n_paste_scar(image, gt, patch_size, mask_max):
    while True:
        mask_size = [random.randint(2, 5), random.randint(2,5), random.randint(5, mask_max)]
        #mask_size = [random.randint(4, 10), random.randint(4, 10), random.randint(10, mask_max)]
        random.shuffle(mask_size)
        mask_pos = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
        mask_pos2 = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
        overlap = [abs(mask_pos[i] -mask_pos2[i]) < mask_size[i]/2 for i in range(len(image.shape))]
        if overlap == [True] * len(image.shape):
            continue
        image[mask_pos2[0]:mask_pos2[0] + mask_size[0], mask_pos2[1]:mask_pos2[1] + mask_size[1], mask_pos2[2]:mask_pos2[2] + mask_size[2]] = image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]]
        gt[mask_pos2[0]:mask_pos2[0] + mask_size[0], mask_pos2[1]:mask_pos2[1] + mask_size[1], mask_pos2[2]:mask_pos2[2] + mask_size[2]] = 1
        return image, gt
  
def copy_n_paste_rot(image, gt, patch_size, mask_max, fg_th = 0.0):
    while True:
        mask_size = [random.randint(10, mask_max) for _ in range(len(image.shape))]
        mask_pos = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
        mask_pos2 = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
        overlap = [abs(mask_pos[i] -mask_pos2[i]) < mask_size[i]/2 for i in range(len(image.shape))]
        if overlap == [True] * len(image.shape):
            continue
        if np.count_nonzero(image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]]) / (mask_size[0]*mask_size[1]*mask_size[2]) < fg_th:
            continue
        rotated_mask = np.zeros_like(gt)
        rotated_mask[mask_pos2[0]:mask_pos2[0] + mask_size[0], mask_pos2[1]:mask_pos2[1] + mask_size[1], mask_pos2[2]:mask_pos2[2] + mask_size[2]] = image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]]
        gt[mask_pos2[0]:mask_pos2[0] + mask_size[0], mask_pos2[1]:mask_pos2[1] + mask_size[1], mask_pos2[2]:mask_pos2[2] + mask_size[2]] = 1
        angle = random.uniform(-45, 45)
        rotated_mask = ndimage.interpolation.rotate(rotated_mask, angle, order= 0, axes = (0,1), reshape = False)
        gt = ndimage.interpolation.rotate(gt, angle, order= 0, axes = (0,1), reshape = False)
        angle = random.uniform(-45, 45)
        rotated_mask = ndimage.interpolation.rotate(rotated_mask, angle, order= 0, axes = (0,2), reshape = False)
        gt = ndimage.interpolation.rotate(gt, angle, order= 0, axes = (0,2), reshape = False)
        angle = random.uniform(-45, 45)
        rotated_mask = ndimage.interpolation.rotate(rotated_mask, angle, order= 0, axes = (1,2), reshape = False)
        gt = ndimage.interpolation.rotate(gt, angle, order= 0, axes = (1,2), reshape = False)

        np.put(image, np.where(gt.ravel()==1), rotated_mask[gt==1])
        return image, gt

def permutation(image, gt, patch_size, mask_max):
    order = [0, 1, 2, 3, 4, 5, 6, 7]
    while order == [0, 1, 2, 3, 4, 5, 6, 7]:
        random.shuffle(order)
    mask_size = [random.randint(10, mask_max) for _ in range(len(image.shape))]
    mask_size = [int(size / 2) * 2 for size in mask_size]
    half_size = [int(size / 2) for size in mask_size]
    mask_pos = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
    ori = np.zeros(mask_size)
    ori = np.copy(image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]])
    for i in range(len(order)):
        if order[i] == i:
            continue
        # Make binary (i - > xyz, ex) 7 -> 111)
        ori_x = i // 4
        ori_y = (i - ori_x * 4) // 2
        ori_z = (i - ori_x * 4 - ori_y * 2) % 2
        ori_x = ori_x * half_size[0]
        ori_y = ori_y * half_size[1]
        ori_z = ori_z * half_size[2]
        
        x = order[i] // 4
        y = (order[i] - x * 4) // 2
        z = (order[i] - x * 4 - y * 2) % 2
        x = mask_pos[0] + x * half_size[0]
        y = mask_pos[1] + y * half_size[1]
        z = mask_pos[2] + z * half_size[2]

        gt[x:x + half_size[0], y:y + half_size[1], z:z + half_size[2]] = 1
        image[x:x + half_size[0], y:y + half_size[1], z:z + half_size[2]] = ori[ori_x:ori_x + half_size[0], ori_y:ori_y + half_size[1], ori_z:ori_z + half_size[2]]
    return image, gt

def sobel_filter(image, gt, patch_size, mask_max):
    mask_size = [random.randint(10, mask_max) for _ in range(len(image.shape))]
    mask_pos = [random.randint(0, patch_size - mask_size[i]) for i in range(len(image.shape))]
    ood = image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]]

    dx = ndimage.sobel(ood, 0)
    dy = ndimage.sobel(ood, 1)
    dz = ndimage.sobel(ood, 2)

    xyz = pow(dx*dx + dy*dy + dz*dz, 1/2)
    xyz /= xyz.max() + 1e-6

    image[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]] = xyz
    gt[mask_pos[0]:mask_pos[0] + mask_size[0], mask_pos[1]:mask_pos[1] + mask_size[1], mask_pos[2]:mask_pos[2] + mask_size[2]] = 1

    return image, gt


# Background remover for GT
def removeBGfromGT(image, aug_image, gt):
    gt[(image == 0) & (aug_image == 0)] = 0
    return gt

def elastic_transform(image, mask, alpha, sigma):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       
       Modified from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    shape = image.shape
    dx = gaussian_filter(( np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(( np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter(( np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    distored_mask = map_coordinates(mask, indices, order=1, mode='reflect')

    return distored_image.reshape(image.shape), distored_mask.reshape(mask.shape)