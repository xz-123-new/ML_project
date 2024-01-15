import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import supervision as sv
from PIL import Image
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

'''
def show_anns(anns):
    if len(anns) == 0:
        return
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    #img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        #print(sum(m))
        color_mask = np.concatenate([np.random.random(3)])
        img[m] = color_mask
    #ax.imshow(img)
    img_jpg = Image.fromarray(img.astype(np.uint8))
    #if img_jpg.mode == 'RGBA':
    #    img_jpg = img_jpg.convert('RGB')
    img_jpg.save('/home/xuzhu/sam/temp/test.jpg')
'''
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
    
sam_checkpoint = "/home/xuzhu/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
i = '/home/xuzhu/sam/data_2/train/img/img0001/slice_001.jpg'
img_path = cv2.imread(i)
img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
image_path = img_path
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

label_path = '/home/xuzhu/sam/RawData/Training/label0001.nni.gz'
#########generate gt point/bbox########


######################
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)
masks = mask_generator.generate(image_path)

masks.shape
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("Houses with segmentation mask")
plt.imshow(image_path)
show_anns(masks)
plt.savefig('/home/xuzhu/sam/temp/test.jpg')
plt.axis('off')

