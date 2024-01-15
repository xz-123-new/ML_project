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
    parser.add_argument('--clackpt',action = 'store_true')
    parser.add_argument('--validate_num', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    test_data, test_labels = data_utils.load_dataset('validation')
    sam = sam_model_registry['vit_h_class'](checkpoint="sam_vit_h_4b8939.pth")
    if args.clackpt:
        sam.load_class_state()
        print("success load ckpt for classifier.")

    sam.to(device='cuda')
    sam.eval()


    print("Begin segment on test dataset")
    test_dices = []
    class_correct = np.zeros(13)
    class_total = np.zeros(13)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(zip(test_data, test_labels)):
            test_labels_binary = (label[None, ...] == np.arange(14)[:, None, None, None]).astype(np.int8)
            print(f"Sample {i}:")
            z_idx_range = list(range(0, data.shape[-1] + 1))
            segment_3d = []

            for idx in tqdm(range(len(z_idx_range) - 1)):
                z_scope = range(z_idx_range[idx], z_idx_range[idx + 1])
                if args.grid_prompt:
                    organ_info, slices_input = data_utils.grid_preporcess(
                        data, label, z_scope, 
                        args.grid_distance
                    )
                else:
                    organ_info, slices_input = data_utils.slice_preporcess(
                        data, label, z_scope, 
                        args.point, args.bbox
                    )
                if len(slices_input) == 0:
                    # gradient is always 0, no backward propagation
                    segment_3d.append(np.zeros_like(label[..., z_scope], dtype=np.int8))
                    torch.cuda.empty_cache()
                    continue
                output = sam(slices_input, multimask_output=False)
                predicted_classes = []
                for (z, organ_class), out in zip(organ_info, output):
                    pred_class = out['class_prediction']
                    gt_class = (torch.tensor(organ_class, dtype = int) - 1).to('cuda')
                    correct += (pred_class.argmax(dim=1) == gt_class).sum().item()
                    total += 1
                    for j in range(13):
                        class_correct[j] += ((pred_class.argmax(dim=1) == gt_class) * (gt_class == j)).sum().item()
                        class_total[j] += (gt_class == j).sum().item()
                    predicted_classes.append((z, 1 + torch.argmax(pred_class, dim=1)))
                segment_result = np.zeros_like(label[..., z_scope], dtype=np.int8)
                for out, (z, organ_class) in zip(output, predicted_classes):
                    out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                    segment_result[:, :, z - z_idx_range[idx]] = (
                        organ_class[:, None, None, None] * out['masks']
                    ).max(dim=0).values.cpu().numpy()[0]
                segment_3d.append(segment_result)
                torch.cuda.empty_cache()
                
            segment_result = np.concatenate(segment_3d, axis=-1)

            dice = data_utils.dice_for_all_classes(segment_result, label)
            dice_dict = {i + 1: dice[i] for i in range(13)}
            # 输出写入文件
            with open('output_test_log.txt', 'a') as f:
                f.write("Dice scores: \n")
                f.write(str(dice_dict) + '\n')
                train_dice = data_utils.all_mdice(dice_dict)
                f.write(f"mDice: {train_dice}\n")
            torch.cuda.empty_cache()

    acc = 0.0
    acc = correct / total
    class_acc = class_correct / class_total
    print(f"Average training mDice: {sum(training_mdice_record) / 20}")
    print("Class accuracy: {}".format(cla_acc.reshape(-1,1)))
    print("accuracy is :{}".format(acc))

