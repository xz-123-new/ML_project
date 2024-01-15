import data_utils
import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import math
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import (ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts)
from typing import List, Tuple, Optional, Union
from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', type=str, nargs='*')
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--grid_prompt', action='store_true')# whether to use grid_prompt 
    parser.add_argument('--grid_distance', type=int, default=16) # to set the distance
    parser.add_argument('--validate_num', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    training_data, training_labels = data_utils.load_dataset('training')

    valid_idx = random.sample(range(1, 25), args.validate_num)
    validation_data = [training_data[i] for i in valid_idx]
    validation_labels = [training_labels[i] for i in valid_idx]

    test_data, test_labels = data_utils.load_dataset('validation')

    sam = sam_model_registry['vit_h_class'](checkpoint='sam_vit_h_4b8939.pth')
    sam.to(device='cuda')

    optimizer = Adam(sam.parameters(), lr=args.lr)
    lr_lambda = lambda epoch: 0.95 ** epoch
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    #lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=10, min_lr=1e-8)

    if args.point is None:
        args.point = []

    training_mdice_record = []
    validation_dices = []
    best_validation_avg_mdice = 0.0
    best_test_mdices = []
    class_correct = np.zeros(13)
    class_total = np.zeros(13)
    correct = 0
    total = 0

    for epoch in range(50):
        print(f"Epoch {epoch + 1}.\n")
        print("Beginning segment on training dataset")
        sam.train()
        training_mdice_record = []
        all_train_loss=0
        for i, (data, label) in enumerate(zip(training_data, training_labels)):
            train_class_label = (label[None, ...] == np.arange(14)[:, None, None, None]).astype(np.int8)
            print(f"Sample {i}:")
            z_idx_range = list(range(0, data.shape[-1] + 1)) 
            segment_3d = []
            loss_for_i = 0.0

            for idx in tqdm(range(len(z_idx_range) - 1)):
                segment_loss = 0.0
                classify_loss = 0.0
                z_scope = range(z_idx_range[idx], z_idx_range[idx + 1])
                if args.grid_prompt:
                    organ_info, slices_input = data_utils.grid_preporcess(data, label, z_scope, args.grid_distance)
                else:
                    organ_info, slices_input = data_utils.slice_preporcess(data, label, z_scope, args.point, args.bbox)
                if len(slices_input) == 0:
                    #####no valid input, skip this slice#########
                    segment_3d.append(np.zeros_like(label[..., z_scope], dtype=np.int8))
                    torch.cuda.empty_cache()
                    continue
                output = sam(slices_input, multimask_output=False)
                num_classes = 0
                predicted_classes = []
                optimizer.zero_grad()
                for (z, organ_class), out in zip(organ_info, output):
                    gt = torch.from_numpy(train_class_label[organ_class, ..., z_scope]).flatten().to('cuda').to(torch.float)
                    pred = out['masks'][:, 0, :, :].flatten()
                    # cross entropy loss of ALL targets on a given slice
                    segment_loss += nn.BCEWithLogitsLoss(reduction='sum')(pred, gt)
                    pred_class = out['class_prediction']
                    gt_class = (torch.tensor(organ_class, dtype=int) - 1).to('cuda') # n_targets
                    # cross entropy loss of class predictions of ALL targets on a given slice
                    classify_loss += nn.CrossEntropyLoss(reduction='sum')(pred_class, gt_class)
                    correct += (pred_class.argmax(dim=1) == gt_class).sum().item()
                    total += 1
                    num_classes += len(organ_class)
                    for j in range(13):
                        class_correct[j] += ((pred_class.argmax(dim=1) == gt_class) * (gt_class == j)).sum().item()
                        class_total[j] += (gt_class == j).sum().item()
                    with torch.no_grad():
                        predicted_classes.append((z, 1 + torch.argmax(pred_class, dim=1)))
                
                segment_loss /= (out['masks'].shape[-1] * out['masks'].shape[-2])
                classify_loss /= num_classes
                slice_loss = segment_loss + classify_loss
                slice_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                with torch.no_grad():
                    loss_for_i += slice_loss.item()
                    segment = np.zeros_like(label[..., z_scope], dtype=np.int8)
                    for out, (z, organ_class) in zip(output, predicted_classes):
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                        segment[:, :, z - z_idx_range[idx]] = (organ_class[:, None, None, None] * out['masks']).max(dim=0).values.cpu().numpy()[0]
                    segment_3d.append(segment)
                torch.cuda.empty_cache()
            
            segment_result = np.concatenate(segment_3d, axis=-1)
            dice = data_utils.dice_for_all_classes(segment_result, label)
            dice_dict = {i + 1: dice[i] for i in range(13)}

            # 输出写入文件
            with open('output_log.txt', 'a') as f:
                f.write("Dice scores: \n")
                f.write(str(dice_dict) + '\n')
                train_dice = data_utils.all_mdice(dice_dict)
                training_mdice_record.append(train_dice)
                f.write(f"mDice: {train_dice}\n")
            torch.cuda.empty_cache()
        acc = 0.0
        acc = correct / total
        class_acc = class_correct / class_total
        print(f"Average training mDice: {sum(training_mdice_record) / 20}")
        print("Class accuracy: {}".format(class_acc.reshape(-1,1)))
        print("accuracy is :{}".format(acc))
        

############################################
########### validation part ################
############################################

        sam.eval()
        with torch.no_grad():
            print("Begin segment on validation dataset")
            validation_dices = []
            class_correct = np.zeros(13)
            class_total = np.zeros(13)
            correct = 0
            total = 0
            best_acc = 0.0
            for i, (data, label) in enumerate(zip(validation_data, validation_labels)):
                validation_b = (label[None, ...] == np.arange(14)[:, None, None, None]).astype(np.int8)
                print(f"Image {i}:")
                z_idx_range = list(range(0, data.shape[-1] + 1))
                segment_3d = []

                for idx in tqdm(range(len(z_idx_range) - 1)):
                    z_scope = range(z_idx_range[idx], z_idx_range[idx + 1])
                    if args.grid_prompt:
                        organ_info, slices_input = data_utils.grid_preporcess(data, label, z_scope, args.grid_distance)
                    else:
                        organ_info, slices_input = data_utils.slice_preporcess(data, label, z_scope, args.point, args.bbox)
                    if len(slices_input) == 0:
                        segment_3d.append(np.zeros_like(label[..., z_scope], dtype=np.int8))
                        torch.cuda.empty_cache()
                        continue
                    output = sam(slices_input, multimask_output=False)
                    predicted_classes = []
                    for (z, organ_class), out in zip(organ_info, output):
                        pred_class = out['class_prediction'] # n_targets * n_classes
                        predicted_classes.append((z, 1 + torch.argmax(pred_class, dim=1)))
                        gt_class = (torch.tensor(organ_class, dtype = int) - 1).to('cuda')
                        correct += (pred_class.argmax(dim=1) == gt_class).sum().item()
                        total += 1
                        for j in range(13):
                            class_correct[j] += ((pred_class.argmax(dim=1) == gt_class) * (gt_class == j)).sum().item()
                            class_total[j] += (gt_class == j).sum().item()
                    segment = np.zeros_like(label[..., z_scope], dtype=np.int8)
                    for out, (z, organ_class) in zip(output, predicted_classes):
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                        segment[:, :, z - z_idx_range[idx]] = (organ_class[:, None, None, None] * out['masks']).max(dim=0).values.cpu().numpy()[0]
                    segment_3d.append(segment)
                    torch.cuda.empty_cache()
                
                segment_result = np.concatenate(segment_3d, axis=-1)

                dice = data_utils.dice_for_all_classes(segment_result, label)
                dice_dict = {i + 1: dice[i] for i in range(13)}

                 # 输出写入文件
                with open('output_valid_log.txt', 'a') as f:
                    f.write(f"Accumulated loss: {loss_for_i}\n")
                    f.write("Dice scores: \n")
                    f.write(str(dice_dict) + '\n')
                    valid_dice = data_utils.all_mdice(dice_dict)
                    validation_dices.append(valid_dice)
                    f.write(f"mDice: {valid_dice}\n")
                torch.cuda.empty_cache()
            avg_validation_mdice = sum(validation_dices) / 4
            print(f"Average validation mDice: {avg_validation_mdice}")
            acc = 0.0
            acc = correct / total
            class_acc = class_correct / class_total
            print("Class accuracy: {}".format(class_acc.reshape(-1,1)))
            print("accuracy is :{}".format(acc))
            if acc > best_acc:
                best_acc = acc
                sam.save_class_state()
                ###torch.save(sam.state_dict(), "classifier_epoch_{}.pth".format(epoch))
                #torch.save(sam.state_dict(), 'fine_tuned_sam_vit_h_with_classification.pth')
                print("Save model at epoch {}".format(epoch))


           