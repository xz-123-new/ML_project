import argparse
import sys
sys.path.append('..')
import json
import torch
import random
import prompt
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from tqdm import tqdm
import utils
import os.path as osp
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Optional, Union

json_path = '../dataset/RawData/dataset_0.json'
data_path = '../dataset/RawData'
batch_size = 1

def get_sam(model_type: str = 'vit_h',
            sam_checkpoint: str = '../sam_vit_h_4b8939.pth',
            device: str = 'cuda:0'):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return sam

def output2result(labels, range_start, used_targets, output):
    segment_result = np.zeros((14, *labels.shape), dtype=np.int8)
    for out, (z, targets) in zip(output, used_targets):
        batched_mask = out['masks'].cpu().numpy()
        segment_result[targets, :, :, z - range_start] = batched_mask[:, 0, :, :]
    return segment_result

def load_dataset(kind):
    with open(json_path) as json_file:
        config = json.load(json_file)
        data_all, label_all = [], []
        print(f'loading {kind}')
        for split_dict in tqdm(config[kind]):
            data = utils.load_data(osp.join(data_path, split_dict['image']))
            data_all.append(data)

            label = utils.load_label(osp.join(data_path, split_dict['label']))
            label_all.append(label)
        return data_all, label_all

if __name__ == '__main__':
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt', type=int, default=0)
    args = parser.parse_args()

    training_data, training_labels = load_dataset('training')
    validation_data, validation_labels = load_dataset('validation')
    sam = utils.get_sam(device=args.device)

    point_prompt = ['center'] #['random', 'center'] , [], []
    bbox_prompt = False # False True True
    bbox_margin = 0 # 0 0 50

    if args.prompt == 1:
        point_prompt = ['random', 'random', 'random', 'random', 'random']
        bbox_prompt = False
        bbox_margin = 0
    elif args.prompt == 2:
        point_prompt = ['random', 'random', 'random', 'random', 'center']
        bbox_prompt = False
        bbox_margin = 0
    elif args.prompt == 3:
        point_prompt = []
        bbox_prompt = True
        bbox_margin = 50


    if bbox_prompt:
        print(f"bounding box prompt with margin {bbox_margin}")
    print(f'point_prompt:{point_prompt}')
    
    total_dice = {}
    for target in range(1, 14):
        total_dice[target] = []   

    print("Start segmentation on training data...")
    for data_id, (data, label) in enumerate(zip(training_data, training_labels)):
        print(f'Sample {data_id}:')
        split_ranges = list(range(0, data.shape[-1], batch_size)) + [data.shape[-1]]
        results = []
        for i in tqdm(range(len(split_ranges) - 1)):
            cur_range = range(split_ranges[i], split_ranges[i + 1])
            used_targets, sam_input = prompt.generate_prompt_input(
                sam, data, label, cur_range, 
                point_prompt, bbox_prompt, bbox_margin
            )
            if sam_input is None or len(sam_input) == 0:
                results.append(np.zeros((14, *label[..., cur_range].shape), dtype=np.int8))
            else:
                sam_output = sam(sam_input, multimask_output=False)
                results.append(output2result(label[..., cur_range], split_ranges[i], used_targets, sam_output))
            torch.cuda.empty_cache()
        results = np.concatenate(results, axis=-1)

        dice_dict = {}
        for target in range(1, 14):
            truth = utils.select_label(label, target)
            if truth.sum() == 0:
                dice_dict[target] = float('nan')
                continue
            pred = results[target]
            dice_dict[target] = utils.dice_score(pred, truth)
        
        print("Dice scores: ")
        print(dice_dict)
        print(f"mDice:  {utils.compute_mdice(dice_dict)}")
        for target in range(1, 14):
            total_dice[target].append(dice_dict[target])
        torch.cuda.empty_cache()

    print("Start segmentation on validation data...")
    for data_id, (data, label) in enumerate(zip(validation_data, validation_labels)):
        print(f'Sample {data_id}:')
        split_ranges = list(range(0, data.shape[-1], batch_size)) + [data.shape[-1]]
        results = []
        for i in tqdm(range(len(split_ranges) - 1)):
            cur_range = range(split_ranges[i], split_ranges[i + 1])
            used_targets, sam_input = prompt.generate_prompt_input(
                sam, data, label, cur_range, 
                point_prompt, bbox_prompt, bbox_margin
            )
            if sam_input is None or len(sam_input) == 0:
                results.append(np.zeros((14, *label[..., cur_range].shape), dtype=np.int8))
            else:
                sam_output = sam(sam_input, multimask_output=False)
                results.append(output2result(label[..., cur_range], split_ranges[i], used_targets, sam_output))
            torch.cuda.empty_cache()
        results = np.concatenate(results, axis=-1)

        dice_dict = {}
        for target in range(1, 14):
            truth = utils.select_label(label, target)
            if truth.sum() == 0:
                dice_dict[target] = float('nan')
                continue
            pred = results[target]
            dice_dict[target] = utils.dice_score(pred, truth)
        
        print("Dice scores: ")
        print(dice_dict)
        print(f"mDice:  {utils.compute_mdice(dice_dict)}")
        for target in range(1, 14):
            total_dice[target].append(dice_dict[target])
        torch.cuda.empty_cache()

    print("Dice scores on each organ:")
    for target in range(1, 14):
        print(f"Organ {target}: {np.mean(total_dice[target])}")



    