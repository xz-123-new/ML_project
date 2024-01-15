import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import math
import json
import os.path 
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional, Union
import torch
from segment_anything.utils.transforms import ResizeLongestSide
import sys

def load_data(img_pth):
    image_obj = nib.load(img_pth)
    image_data = image_obj.get_fdata()
    low, upper = image_data.min(), image_data.max()
    image_data = (image_data - low) / (upper - low) * 256
    return np.repeat(np.expand_dims(image_data, axis=2),3,axis=2).astype(np.uint8)

def load_label(label_path):
    label_obj = nib.load(label_path)
    label_data = label_obj.get_fdata()
    return label_data.astype(np.int8)

def label_preprocess(label):
    class_expand_label = label[None,...]
    return (class_expand_label == np.arange(14)[:, None, None, None]).astype(np.int8)

def load_dataset(split = 'training'):
    data_dir = '/home/xuzhu/sam/RawData'
    with open('/home/xuzhu/sam/RawData/config.json') as json_file:
        config = json.load(json_file)
        split_data = []
        split_labels = []
        for item in config[split]:
            split_data.append(load_data(os.path.join(data_dir, item['image'])))
            split_labels.append(load_label(os.path.join(data_dir, item['label'])))
        print(f'finish load {split} data.\n')
        return split_data, split_labels

def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    i_2 = (pred & gt).sum() * 2
    ##########double it for pretty number according to dice definition
    u = pred.sum() + gt.sum()
    return i_2 / u

def segment_3d(labels, z_idx, classes_info, output):
    segment_3d = np.zeros((14, *labels.shape), dtype=np.int8)
    for out, (z, classes) in zip(output, classes_info):
        slice_mask = out['masks'].cpu().numpy()
        segment_3d[classes, :, :, z - z_idx] = slice_mask[:, 0, :, :]
    return segment_3d

def mdice_across_class(dice_dict,epmty_idx):
    count = 0 
    sum = 0
    for val in dice_dict.values():
        if not val==999999: 
            count += 1
            sum += val
    return sum / count
    
    
def generate_prompt(input_label,point,bbox):
    if input_label.sum() == 0: 
        return None, None, None
    if 'center' in point:  #############could be change to more type,i.e., random points or multiple points in task_1, but center point is best in practical, so only adopt center point######### 
        h,w = input_label.shape[0],input_label.shape[1]
        dist = ndimage.distance_transform_edt(input_label)
        val_max, idx_max = dist.max(), dist.argmax()
        center_point = np.array([idx_max % h, idx_max // w], dtype=np.int32)[None,:]
    elif 'multiple' in point:
        #############we always use center_point name to represent our point prompt for simplisity
        h,w = input_label.shape[0],input_label.shape[1]
        indices_of_ones = np.argwhere(input_label == 1)
        try:
            random_sample_indices = np.random.choice(indices_of_ones.shape[0], size=5, replace=False)
            random_sample_points = indices_of_ones[random_sample_indices]
            center_point = random_sample_points
        except:
            center_point = np.ones((0, 2), dtype=np.int64)
    elif 'multiple_w_center':
        h,w = input_label.shape[0],input_label.shape[1]
        dist = ndimage.distance_transform_edt(input_label)
        val_max, idx_max = dist.max(), dist.argmax()
        center_point_origin = np.array([idx_max % h, idx_max // w], dtype=np.int32)[None,:]
        indices_of_ones = np.argwhere(input_label == 1)
        random_sample_indices = np.random.choice(indices_of_ones.shape[0], size=5, replace=False)
        random_sample_points = indices_of_ones[random_sample_indices]
        center_point = np.vstack((center_point_origin,random_sample_points))
    else:
        center_point = np.ones((0, 2), dtype=np.int64)
    center_label = np.ones((center_point.shape[0], ), dtype=np.int8)
    if center_point.shape[0] == 0:
        center_point = None
        center_label = None
    if not bbox is None:
        rows, columns = np.where(input_label)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = columns.min(), columns.max()
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)
    else:
        bbox = None
    ########temporaly for debug
    bbox = None
    return center_point, center_label, bbox
 
def generate_slices_info(input_label,point,bbox,classes= None):
    classes = list(range(1, 14))
    ######we only exert on all classes
    slices_coords = []
    slices_labels = []
    slices_boxes = []
    slices_classes = []
    #########collect all prompta for each class############
    for organ_class in classes:
        selected_label = (input_label == organ_class).astype(np.int8)
        coords, labels, bbox = generate_prompt(selected_label,point,bbox)
        ######skip no prompts class########3
        if coords is None and labels is None and bbox is None:
            #raise ValueError('Shold exist at least one kind of prompt.')
            continue
        ######## only exert on class with prompts#########
        slices_classes.append(organ_class)
        if coords is not None:
            slices_coords.append(coords)
        if labels is not None:
            slices_labels.append(labels)
        if bbox is not None:
            slices_boxes.append(bbox)
    return slices_classes, slices_coords, slices_labels, slices_boxes  


def slice_preporcess(data, labels ,z_batch_range,point,bbox,classes=None):
    transform = ResizeLongestSide(512)
    organ_class_list = []
    slices_input = []

    for i in z_batch_range:
        image_i = torch.from_numpy(transform.apply_image(data[..., i])).to('cuda').permute(2, 0, 1).contiguous()
        gen_classes, slice_coords, slice_labels, slice_box = generate_slices_info(labels[..., i], point, bbox,classes)
        if len(slice_coords) > 0:
            slice_coords = np.stack(slice_coords, axis=0)
        else:
            slice_coords = None
        if len(slice_labels) > 0:
            slice_labels = np.stack(slice_labels, axis=0)
        else:
            slice_labels = None
        if len(slice_box) > 0:
            slice_box = np.stack(slice_box, axis=0)
        else:
            slice_box = None
        if slice_coords is None and slice_labels is None and slice_box is None:
            ########jump to next z-idx
            continue
        organ_class_list.append((i, gen_classes))
        input_dict = {}
        input_dict['image'] = image_i
        input_dict['original_size'] = data.shape[:2]
        if slice_box is not None:
            slice_box = transform.apply_boxes_torch(torch.from_numpy(slice_box).to('cuda'), data.shape[:2])
            input_dict['boxes'] = slice_box
        if slice_coords is not None:
            slice_coords = transform.apply_coords_torch(torch.from_numpy(slice_coords).to('cuda'), data.shape[:2])
            input_dict['point_coords'] = slice_coords
        if slice_labels is not None:
            slice_labels = torch.from_numpy(slice_labels).to('cuda')
            input_dict['point_labels'] = slice_labels
        slices_input.append(input_dict)
    return organ_class_list, slices_input


