import data_utils
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import logging
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from segment_anything import sam_model_registry, SamPredictor
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']='6'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', type=str, nargs='*')
    parser.add_argument('--bbox', action='store_true')
    #parser.add_argument('--bbox_margin', type=int, default=0,)
    parser.add_argument('--validate_num', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    writer = SummaryWriter(log_dir='./single_3')
    training_data, training_labels = data_utils.load_dataset('training')
    #######random draw some in training set as validation set
    valid_idx = random.sample(range(1, 25), args.validate_num)
    validation_data = [training_data[i] for i in valid_idx]
    validation_labels = [training_labels[i] for i in valid_idx]

    training_data = [training_data[i] for i in range(24) if i not in valid_idx]
    training_labels = [training_labels[i] for i in range(24) if i not in valid_idx]
    ######validation serve as test set
    test_data, test_labels = data_utils.load_dataset('validation')

    sam = sam_model_registry['vit_h'](checkpoint='/home/xuzhu/sam/sam_vit_h_4b8939.pth')
    sam.to(device='cuda')
    for param in sam.parameters():
        param.requires_grad = False
    ##########  we only finetune the decoder part of sam  ################
    trainable_params = list(sam.mask_decoder.parameters())
    for param in trainable_params:
        param.requires_grad = True
    # for param in sam.parameters():
    #      param.requires_grad = True
    # trainable_params = sam.parameters()
    for name in trainable_params:
        print(name)
    #criterion = nn.CrossEntropyLoss()
    optimizer = SGD(trainable_params, lr=0.001, momentum=0.9)
    lr_lambda = lambda epoch: 0.95 ** epoch
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    clip_value = 10
    
    if args.point is None:
        args.point = []
    ##########for store the best ckpt############
    training_dices = []
    validation_dices = []
    best_val_mdice_average_on_split = 0
    best_test_mdices = []
    best_test_mdice_average_on_split = 0
    #######################   training #################
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}.\n")
        all_train_loss=0
        sam.train()
        training_mdice_record = []
        ############enumerate on all training samples to finetune################
        for i, (data, label) in enumerate(zip(training_data, training_labels)):
            print(f"Image {i}:")
            train_class_label = data_utils.label_preprocess(label)
            z_idx_range = list(range(0, data.shape[-1] + 1)) 
            ########resume for dice computing
            segment_3d = []
            #####record loss for each 3d image
            loss_for_i = 0.0
             
            for idx in tqdm(range(len(z_idx_range) - 1)):
                slice_loss = 0.0
                
                #since batch_size is not implemented, z_scope actually stand for each slice
                z_scope = range(z_idx_range[idx], z_idx_range[idx + 1])
                organ_info, slices_input = data_utils.slice_preporcess(data, label, z_scope, 
                    args.point, args.bbox)
                if len(slices_input) == 0:
                    #####no valid input, skip this slice#########
                    segment_3d.append(np.zeros((14, *label[..., z_scope].shape), dtype=np.int8))
                    torch.cuda.empty_cache()
                    continue
                output = sam(slices_input, multimask_output=False)
                optimizer.zero_grad()
                #output[0]['masks'].requires_grad = True
                for (z, organ_class), out in zip(organ_info, output):
                    gt = torch.from_numpy(train_class_label[organ_class, ..., z_scope]).flatten().to('cuda').to(torch.float)
                    pred = out['masks'][:, 0, :, :].flatten().float()
                    pred.requires_grad = True
                    ###transfer pred to float dtype to ensure loss backprop
                    #slice_loss += nn.BCELoss(reduction='sum')(pred, gt)
                    slice_loss += nn.BCEWithLogitsLoss(reduction='sum')(pred, gt)
                ##################average on h*w ##################
                slice_loss /= (out['masks'].shape[0] * out['masks'].shape[1])
                slice_loss.backward()
                for param in trainable_params:
                    if param.grad is not None:
                        print(f"Gradient of {name}: {param.grad}")
                optimizer.step()
                lr_scheduler.step()
                torch.nn.utils.clip_grad_norm_(trainable_params, clip_value)
                with torch.no_grad():
                    loss_for_i += slice_loss.item()
                    for out in output:
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                    segment = data_utils.segment_3d(label[..., z_scope],  z_idx_range[idx],organ_info, output)
                    segment_3d.append(segment)
                torch.cuda.empty_cache()
            segment_result = np.concatenate(segment_3d, axis=-1)
            #####compute dice for each class##########3
            dice_dict = {}
            empty_idx = []
            for organ_class in range(1, 14):
                gt = (label == organ_class).astype(np.int8)    
                if gt.sum() == 0:
                    dice_dict[organ_class] = 999999
                    empty_idx.append(organ_class)
                    continue
                pred = segment_result[organ_class]
                dice_dict[organ_class] = data_utils.dice_score(pred, gt)
            with open('./sinlgle_log_3.txt','a') as f:
                writer.add_scalar(f"Loss_{i}_image",loss_for_i, epoch)
                f.write(f"Loss for {i}^th image is : {loss_for_i}, the dice in training split is {dice_dict}.\n")
                train_dice = data_utils.mdice_across_class(dice_dict,empty_idx)
                training_mdice_record.append(train_dice)
                f.write(f"the mDice across all organ class for image {i} is :  {train_dice}.\n")
            writer.add_scalar(f"Loss_for_whole_train_set",all_train_loss, epoch)
            all_train_loss +=loss_for_i
            torch.cuda.empty_cache()
            ############average on all training 3d images
        #writer.add_scalar()
        with open('./sinlgle_log_3.txt','a') as f:
            f.write(f"Training split mDice across all images in epoch{epoch}: {sum(training_mdice_record) / 21}.\n")
        f.close()
        writer.add_scalar("mdice_train_set",sum(training_mdice_record) / 21,epoch)
        #######################   validating #################
        sam.eval()
        with torch.no_grad():
            validation_dices = []
            for i, (data, label) in enumerate(zip(validation_data, validation_labels)):
                print(f"Image {i}:")
                z_idx_range = list(range(0, data.shape[-1] + 1)) 
                segment_3d = []
                for idx in tqdm(range(len(z_idx_range) - 1)):
                    z_scope = range(z_idx_range[idx], z_idx_range[idx + 1])
                    organ_info, slices_input = data_utils.slice_preporcess(data, label, z_scope, 
                        args.point, args.bbox)
                    if len(slices_input) == 0:
                        segment_3d.append(np.zeros((14, *label[..., z_scope].shape), dtype=np.int8))
                        torch.cuda.empty_cache()
                        continue
                    output = sam(slices_input, multimask_output=False)
                    for out in output:
                        out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                    segment = data_utils.segment_3d(label[..., z_scope], 
                                                    z_idx_range[idx],
                                                    organ_info, output)
                    segment_3d.append(segment)
                    torch.cuda.empty_cache()
                segment_result = np.concatenate(segment_3d, axis=-1)
                dice_dict = {}
                empty_idx = []
                for organ_class in range(1, 14):
                    gt = (label == organ_class).astype(np.int8)
                    if gt.sum() == 0:
                        dice_dict[organ_class] = 999999
                        empty_idx.append(organ_class)
                        continue
                    pred = segment_result[organ_class]
                    dice_dict[organ_class] = data_utils.dice_score(pred, gt)

                with open('./sinlgle_log_3.txt','a') as f:
                    #f.write(dice_dict)
                    f.write(f"The dice in validation split is {dice_dict}.\n")
                    valid_mdice = data_utils.mdice_across_class(dice_dict,organ_class)
                    validation_dices.append(valid_mdice)
                    f.write(f"mDice across all organ class for val image {i} is :  {valid_mdice}.\n")
                f.close()
                writer.add_scalar(f"mdice_for_{i}_val_image",valid_mdice,epoch)
                torch.cuda.empty_cache()
            ############ we randomly adopt 3 image as validation split, should change the number accordingly##########
            val_mdice_this_epoch = sum(validation_dices) / 3
            with open('./sinlgle_log_3.txt','a') as f:
                f.write(f"Validation mDice across all val imges in epoch {epoch} is : {val_mdice_this_epoch}.\n")
            f.close()
            writer.add_scalar("mdice_val_set",val_mdice_this_epoch,epoch)
            #######################  testing #################
            ########### we only exert test on validation set when validation performance is updated.
            if val_mdice_this_epoch > best_val_mdice_average_on_split:
                best_val_mdice_average_on_split = val_mdice_this_epoch
                # best_test_mdices = test_dices
                # best_test_mdice_average_on_split = sum(test_dices)/len(test_dices)
                # torch.save(sam.state_dict(), '/home/xuzhu/sam/ckpt/best_fine_tuned_sam_vit_h.pth')
                # with open('./sinlgle_log_3.txt','a') as f:
                #     f.write(f'the best test_mdice for now is {best_test_mdice_average_on_split}.\n')
                # f.close()
                test_dices = []
                for i, (data, label) in enumerate(zip(test_data, test_labels)):
                    print(f"Image {i}:")
                    z_idx_range = list(range(0, data.shape[-1] + 1)) 
                    segment_3d = []
                    for idx in tqdm(range(len(z_idx_range) - 1)):
                        z_scope = range(z_idx_range[idx], z_idx_range[idx + 1])
                        organ_info, slices_input = data_utils.slice_preporcess(data, label, z_scope, 
                            args.point, args.bbox)
                        if len(slices_input) == 0:
                            segment_3d.append(np.zeros((14, *label[..., z_scope].shape), dtype=np.int8))
                            torch.cuda.empty_cache()
                            continue
                        output = sam(slices_input, multimask_output=False)
                        for out in output:
                            out['masks'] = (out['masks'].detach() > sam.mask_threshold).to(torch.int8)
                        segment = data_utils.segment_3d(label[..., z_scope], 
                                                        z_idx_range[idx],
                                                        organ_info, output)
                        segment_3d.append(segment)
                        torch.cuda.empty_cache()
                    segment_result = np.concatenate(segment_3d, axis=-1)
                    #####compute dice for each class##########
                    dice_dict = {}
                    empty_idx = []
                    for organ_class in range(1, 14):
                        gt = (label == organ_class).astype(np.int8)
                        if gt.sum() == 0:
                            dice_dict[organ_class] = 999999
                            empty_idx.append(organ_class)
                            continue
                        pred = segment_result[organ_class]
                        dice_dict[organ_class] = data_utils.dice_score(pred, gt)
                    with open('./sinlgle_log_3.txt','a') as f:
                        f.write(f'dice score for test image {i} is {dice_dict}.\n')
                        test_dice = data_utils.mdice_across_class(dice_dict,empty_idx)
                        test_dices.append(test_dice)
                        f.write(f"mDice across all organ class in test set is in epoch {epoch} is : {test_dice}.\n")
                    f.close()
                    torch.cuda.empty_cache()
                    writer.add_scalar(f"mdice_for_{i}_test_image",test_dice,epoch)
                best_test_mdices = test_dices
                best_test_mdice_average_on_split = sum(test_dices)/len(test_dices)
                writer.add_scalar("Test_mdice",best_test_mdice_average_on_split,epoch)
                torch.save(sam.state_dict(), '/home/xuzhu/sam/ckpt/best_fine_tuned_sam_vit_h.pth')
                with open('./sinlgle_log_3.txt','a') as f:
                    f.write(f'the best test_mdice for now is {best_test_mdice_average_on_split}.\n')
                f.close()
    with open('./sinlgle_log_3.txt','a') as f:
        f.write(f'the best test_mdice among finetuning is {best_test_mdice_average_on_split}.\n')
        f.write(f'the corresponding test_mdice for each test image is {best_test_mdices}.\n')
    f.close()


