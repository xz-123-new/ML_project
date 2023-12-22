import torch
import numpy as np
import cv2
import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import supervision as sv
from PIL import Image
import sys
import nibabel as nib
import random
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    

def my_dice(pr, gt, eps=1e-7):
    pr = torch.tensor(pr)
    gt = torch.tensor(gt)
    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr)
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()




def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))    

def gen_gt_info(label,category,slice):
    label_slice = label[:,:,slice]
    indices = np.where(label_slice == category)
    random_index = random.choice(range(len(indices[0])))

    single_point = np.array([[indices[0][random_index],indices[1][random_index]]])
    min_row, min_col = np.min(indices, axis=1)
    max_row, max_col = np.max(indices, axis=1)
    mid_row = (min_row + max_row)/2
    mid_col = (min_col + max_col)/2
    center_pt = np.array([[mid_row,mid_col]])

    bounding_box =  np.array([[min_row,min_col,max_row,max_col]])
    gt_for_eval = (label_slice == category).astype(int)
    return bounding_box, single_point,center_pt, gt_for_eval 
        
sam_checkpoint = "/home/xuzhu/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

i = '/home/xuzhu/sam/data_2/train/img/img0001/slice_001.jpg'
img_path = cv2.imread(i)
img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
image_path = img_path

#print(np.mean(np.load('/home/xuzhu/sam/temp/all_dice_list.npy')))


all_dice_val = []
for data_idx in range(1,11): 
    label_path = os.path.join("/home/xuzhu/sam/RawData/Training/label/label"+f"{data_idx:0{4}}.nii.gz")
    #########generate gt point/bbox########
    label = nib.load(label_path).get_fdata()
    slice = label.shape[2]
    for i in range(1,14):#target category
        predicts = []
        gt_labels = []
        for j in range(slice):
            if i in label[:,:,j]:
                #print('i is {}',i)
                #print('j is {}',j)
                box, random_single_pt,center_pt, gt_for_eval = gen_gt_info(label,i,j)
                input_bbox = box
                #input_point = random_single_pt 
                input_point = center_pt
                image_path = os.path.join('/home/xuzhu/sam/data_2/train/img/img'+f'{data_idx:0{4}}/slice_'+f"{j:0{3}}.jpg")
                image = cv2.imread(image_path)
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=np.array([1]),
                    #box=input_bbox,
                    multimask_output=False,
                )
                predict = masks.astype(int)[0]
                gt_labels.append(gt_for_eval)
                predicts.append(predict)
                #plt.figure(figsize=(10,10))
                #plt.imshow(image)
                #show_mask(masks, plt.gca())
                #show_points(input_point, plt.gca())
                #plt.savefig('/home/xuzhu/sam/temp/test_2.jpg')
                #plt.axis('off')
                #plt.show()
        predicts = torch.Tensor(predicts)
        gt_labels = torch.Tensor(gt_labels)     
        dice_val  = my_dice(predicts,gt_labels) 
        all_dice_val.append(dice_val)
        print(f'finish category{i} for data_idx {data_idx}.\n')
mDice = np.mean(all_dice_val)
print(f'the mDice is {mDice}')
save_all_dice = np.array(all_dice_val)
file_path = '/home/xuzhu/sam/temp/all_dice_list_center_pt.npy'
np.save(file_path, save_all_dice)