import os
import argparse
import numpy as np
import nibabel as nib

from scipy.ndimage import zoom

from networks import *
from utils import *

def inference_brain(model, device, test_img, input_path, output_path, eval):
    model.eval()
    patch_size = 64

    img_orig = nib.load(os.path.join(input_path, test_img))
    target_size = patch_size
    resized_img, orientation = resize_brain(img_orig, target_size)   
    
    # img : B x C x X x Y x Z
    img = np.array(resized_img, dtype = np.float32)
    img= np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = 0)
    img = torch.tensor(img).to(device)
    
    with torch.no_grad():
        S_img, pred_img_cls = model(img.float())
        pred_img_cls = torch.flatten(pred_img_cls)

    S_img = S_img.to('cpu').numpy()
    pred_img_cls = pred_img_cls.to('cpu').numpy()

    cls_img = pred_img_cls[0]

    if eval == 'sample':
        f = open(os.path.join(output_path, test_img + '.txt'), 'w')
        f.write(str(cls_img))
        f.close()
        
    elif eval == 'pixel':
        pixel = recon(S_img[0,0,:,:,:], img_orig.shape, orientation)
        nii_img = nib.Nifti1Image(pixel, img_orig.affine, img_orig.header)
        nib.save(nii_img, os.path.join(output_path, test_img))



def inference_abdom(model, device, test_img, input_path, output_path, eval):
    model.eval()
    patch_size = 128
    stride = int(patch_size/2)

    img_orig = nib.load(os.path.join(input_path, test_img))
    target_size = patch_size * 2
    resized_img, orientation = resize_abdom(img_orig, target_size)   

    img = np.array(resized_img, dtype = np.float32)
    img= np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = 0)
    img = torch.tensor(img)

    # img : B x C x X x Y x Z
    S_img = np.zeros_like(img)
    cls_img = 0
    overlapped_cnt = np.zeros_like(img)
    
    for position in range (27):
        x = position // 9
        y = (position - 9*x) // 3
        z = (position - 9*x - 3*y) % 3
        x = stride * x
        y = stride * y
        z = stride * z

        img_patch = img[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size]
        overlapped_cnt[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size] += 1.0
        
        label = np.zeros((1, 27, patch_size, patch_size, patch_size))
        label[:, position, :, :, :] = 1
        img_patch = np.concatenate((img_patch, label), axis=1)
        img_patch = torch.from_numpy(img_patch).to(device)
        
        with torch.no_grad():
            S_img_patch, pred_img_cls = model(img_patch.float())
            pred_img_cls = torch.flatten(pred_img_cls)

        S_img_patch = S_img_patch.to('cpu').numpy()
        pred_img_cls = pred_img_cls.to('cpu').numpy()

        S_img[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size] += S_img_patch
        if cls_img  < pred_img_cls[0]:
            cls_img = pred_img_cls[0]

    S_img = S_img / overlapped_cnt


    if eval == 'sample':
        f = open(os.path.join(output_path, test_img + '.txt'), 'w')
        f.write(str(cls_img))
        f.close()
        
    elif eval == 'pixel':
        pixel = recon(S_img[0,0,:,:,:], img_orig.shape, orientation)
        nii_img = nib.Nifti1Image(pixel, img_orig.affine, img_orig.header)
        nib.save(nii_img, os.path.join(output_path, test_img))



if __name__ == "__main__":
    # Training args
    parser = argparse.ArgumentParser(description='#### MOOD-CGV TEAM ####')
    parser.add_argument('--d', type=str, default='brain',
                        help='choose the data between brain and abdom')
    parser.add_argument('--e', type=str, default='sample', metavar='N',
                        help='choose the evaluation between sample and pixel')
    parser.add_argument('--i', type=str, default='/mnt/data', metavar='N',
                        help='Input directory')
    parser.add_argument('--o', type=str, default='/mnt/pred', metavar='N',
                        help='Input directory')    
    args = parser.parse_args()

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    if args.d == 'brain':
        model = Unet_brain(num_channels=64).to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('/workspace/weights/mood_brain_weight.pth'))
    
    elif args.d == 'abdom':
        model = Unet_abdom(num_channels=64).to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('/workspace/weights/mood_abdom_weight.pth'))

    test_imgs = os.listdir(args.i)

    for test_img in test_imgs :
        if args.d == 'brain':
            inference_brain(model, device, test_img, args.i, args.o, args.e)
        elif args.d == 'abdom':
            inference_abdom(model, device, test_img, args.i, args.o, args.e)
