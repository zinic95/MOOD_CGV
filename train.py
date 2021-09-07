import os
import time
import argparse
import numpy as np
import logging
import random
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Segloss

from dataset import MOODDataset
from networks import *

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train(model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls, optimizer, scheduler, patch_size, lamda):
    model.train()
    optimizer.zero_grad()

    correct = 0.0

    img_patch, imgGT_patch, img_cls = img_patch.to(device), imgGT_patch.to(device), img_cls.to(device)
    aug_patch, augGT_patch, aug_cls = aug_patch.to(device), augGT_patch.to(device), aug_cls.to(device)

    pred_normal_seg, pred_normal_cls = model(img_patch.float())
    pred_aug_seg, pred_aug_cls = model(aug_patch.float())
    pred_normal_cls = torch.flatten(pred_normal_cls)
    pred_aug_cls = torch.flatten(pred_aug_cls)

    # calculate the loss
    bceLoss = nn.BCELoss()
    normal_seg_loss = bceLoss(pred_normal_seg, imgGT_patch.float())
    aug_seg_loss = bceLoss(pred_aug_seg, augGT_patch.float())
    seg_loss = normal_seg_loss + aug_seg_loss
    normal_cls_loss = bceLoss(pred_normal_cls, img_cls.float())
    aug_cls_loss = bceLoss(pred_aug_cls, aug_cls.float())
    cls_loss = normal_cls_loss + aug_cls_loss

    # aviod the trivial solution
    _, FN = Segloss(pred_aug_seg, augGT_patch.float(), patch_size)

    total_loss = seg_loss + FN * lamda + cls_loss

    logging.info(f"Cls BCE Loss: {cls_loss.item():.6f} | Seg BCE Loss: {seg_loss.item():.6f} | FN loss {FN.item() * lamda:.6f}")

    for i in range (pred_normal_cls.shape[0]):
        if pred_normal_cls[i].round() == img_cls[i]:
            correct += 0.5
        if pred_aug_cls[i].round() == aug_cls[i]:
            correct += 0.5
  
    # optimize the parameters
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    return total_loss.item(), correct

def test(model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls):
    torch.cuda.empty_cache()
    model.eval()

    correct = 0.0
    img_patch, imgGT_patch, img_cls = img_patch.to(device), imgGT_patch.to(device), img_cls.to(device)
    aug_patch, augGT_patch, aug_cls = aug_patch.to(device), augGT_patch.to(device), aug_cls.to(device)

    with torch.no_grad():
        pred_normal_seg, pred_normal_cls = model(img_patch.float())
        pred_aug_seg, pred_aug_cls = model(aug_patch.float())
        pred_normal_cls = torch.flatten(pred_normal_cls)
        pred_aug_cls = torch.flatten(pred_aug_cls)

   # calculate the loss
    bceLoss = nn.BCELoss()
    normal_seg_loss = bceLoss(pred_normal_seg, imgGT_patch.float())
    aug_seg_loss = bceLoss(pred_aug_seg, augGT_patch.float())
    seg_loss = normal_seg_loss + aug_seg_loss

    normal_cls_loss = bceLoss(pred_normal_cls, img_cls.float())
    aug_cls_loss = bceLoss(pred_aug_cls, aug_cls.float())
    cls_loss = normal_cls_loss + aug_cls_loss

    total_loss = seg_loss + cls_loss
    logging.info(f"*TEST* Cls BCE Loss: {cls_loss.item():.6f} | Seg BCE Loss: {seg_loss.item():.6f}")
    
    
    for i in range (pred_normal_cls.shape[0]):
        if pred_normal_cls[i].round() == img_cls[i]:
            correct += 0.5
        if pred_aug_cls[i].round() == aug_cls[i]:
            correct += 0.5

    torch.cuda.empty_cache()
    return seg_loss.item(), cls_loss.item(), total_loss.item(), correct
    

if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)

    # Training args
    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--dataset', type=str, default='/root/MOOD2021/dataset/brain_train',
                        help='path of processed dataset')
    parser.add_argument('--weight', type=str, default='./weights',
                        help='path of the weights folder')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epoches', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of iterations to log (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--category', type=str, default='brain', metavar='N',
                        help='Select the category brain or abdom (defualt brain)')
    parser.add_argument('--patch-size', type=int, default=128, metavar='N',
                        help='patch-size (default=128')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set random seed for reproducibility
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Torch use the device: %s" % device)

    if args.category == 'brain':
        model = Unet_brain(num_channels=64).to(device)
        patch_size = 64
    elif args.category == 'abdom':
        model = Unet_abdom(num_channels=64).to(device)
        patch_size = 128
    else:
        logging.info("Choose the Correct Category")

    model = nn.DataParallel(model)
 
    batch_size = args.batch_size
    train_dataset = MOODDataset(args.dataset, subset='train', category = args.category)
    test_dataset = MOODDataset(args.dataset, subset='valid', category = args.category)
   
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=generator)
    train_iterator = iter(train_loader)

    total_iteration = args.epoches * len(train_loader)
    train_interval = args.log_interval * len(train_loader) 

    logging.info(f"total iter: {total_iteration}")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    iteration = 1
    best_train_loss, best_test_loss = float('inf'), float('inf')

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*10, steps_per_epoch=len(train_loader), epochs=args.epoches, anneal_strategy='linear')
 
    epoch_train_loss = []
    epoch_test_loss = []
    epoch_test_segloss = []
    epoch_test_clsloss = []
    correct_train_count = 0
    correct_test_count = 0
    epoch_train_accuracy = 0.
    epoch_test_accuracy = 0.
    v_clsloss = 0.
    start_time = time.time()

    # Seed initializaiton for patch reproducibility
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    while iteration <= total_iteration:
        # 0 epoch: 1.0 ~ max epoch* 0.3: 0.05 (linear)
        lamda = 100 * (1.0 - iteration /(args.epoches * 0.3 * len(train_loader)))
        lamda = 0.05 if lamda < 0 else lamda

        try:
            # Samples the batch
            img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls = next(train_iterator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_iterator = iter(train_loader)
            img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls = next(train_iterator)
    
        t_total_loss, t_correct = train(model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls, optimizer, scheduler, patch_size, lamda)

        if (iteration % train_interval == 0):
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            epoch_train_accuracy = (correct_train_count / (args.log_interval * len(train_dataset))) * 100

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval}: \t Loss: {avg_train_loss:.6f}\t')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                logging.info(f'--- Saving model at Avg Train Loss:{avg_train_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weight, './mood_best_train_' + args.identifier +'.pth'))

            # validation process
            test_iterator = iter(test_loader)
            for i in range(len(test_loader)):
                img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls = next(test_iterator)

                v_segloss, v_clsloss, v_total_loss, v_correct = test(model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls)

                epoch_test_loss.append(v_total_loss)
                epoch_test_segloss.append(v_segloss)
                epoch_test_clsloss.append(v_clsloss)
                correct_test_count += v_correct

            avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
            avg_test_segloss = sum(epoch_test_segloss) / len(epoch_test_segloss)
            avg_test_clsloss = sum(epoch_test_clsloss) / len(epoch_test_clsloss)
            epoch_test_accuracy = (correct_test_count / len(test_dataset)) * 100

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval} eval: \t Loss: {avg_test_loss:.6f}\t')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                logging.info(f'--- Saving model at Avg Valid Loss:{avg_test_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weight, './mood_best_valid_' + args.identifier +'.pth'))

            # save snapshot for resume training
            logging.info('--- Saving snapshot ---')
            torch.save({
                'iteration': iteration+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_train_loss': best_train_loss,
                'best_test_loss': best_test_loss,
            },
                os.path.join(args.checkpoints, 'latest_checkpoints_' + args.identifier +'.pth'))

            logging.info(f"--- {time.time() - start_time} seconds ---")

            epoch_train_loss = []
            epoch_test_loss = []
            epoch_test_segloss = []
            epoch_test_clsloss = []
            correct_train_count = 0
            correct_test_count = 0
            start_time = time.time()
        iteration += 1