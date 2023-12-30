import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import sys
sys.path.append('..')
from segment_anything import sam_model_registry, SamPredictor
import math

def load_label(label_path: str) -> np.ndarray:
    """
    Load file `label_path` as a 3D array of integer labels.
    """
    label_obj = nib.load(label_path)
    label_data: np.ndarray = label_obj.get_fdata()
    return label_data.astype(np.int8)

def select_label(full_label: np.ndarray, fg_label: int) -> np.ndarray:
    """
    Set the elements with value `fg_label` in `full_label` to 1, and
    others to 0. 
    """
    return (full_label == fg_label).astype(np.int8)

def find_center(selected_label: np.ndarray) -> np.ndarray:
    """
    Find the coordinate of center point of the object in slice `selected_label`,
    in the form (X, Y) with shape (2,). Raise `ValueError` if there is no object 
    in the slice.
    """
    w = selected_label.shape[1]
    dist = ndimage.distance_transform_edt(selected_label)
    max_, argmax_ = dist.max(), dist.argmax()
    if int(max_) == 0:
        raise ValueError("The image slice is empty!")
    return np.array([argmax_ % w, argmax_ // w], dtype=np.int32)

def find_bounding_box(selected_label: np.ndarray, margin: int = 0) -> np.ndarray:
    """
    Find the bounding box of the object in slice `selected_label`, in XYXY format,
    with shape (4,). If `margin > 0`, the box will be larger than the bounding box 
    by `margin` at four sides. Raise `ValueError` if there is no object in the slice.
    """
    rows, columns = np.where(selected_label)
    try:
        y_min, y_max = rows.min(), rows.max() + 1
        x_min, x_max = columns.min(), columns.max() + 1
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(selected_label.shape[1], x_max + margin)
        y_max = min(selected_label.shape[0], y_max + margin)
        return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)
    except:
        raise ValueError("The image slice is empty!")

def find_fg_random(selected_label: np.ndarray, num_points: int = 1) -> np.ndarray:
    """
    Generate `num_points` random points in the foreground of slice `selected_label`,
    in shape (num_points, 2). Raise `ValueError` if there is no foreground.

    NOTE: This method doesn't ensure the point coordinates are different.
    """
    rows, columns = np.where(selected_label)
    try:
        indices = np.random.randint(rows.shape[0], size=(num_points,))
        return np.stack([columns[indices], rows[indices]], axis=0).transpose()
    except:
        raise ValueError("The image slice is empty!")   

def find_bg_random(selected_label: np.ndarray, num_points: int = 1) -> np.ndarray:
    """
    Generate `num_points` random points in the foreground of slice `selected_label`,
    in shape (num_points, 2). Raise `ValueError` if there is no foreground.

    NOTE: This method doesn't ensure the point coordinates are different.
    """
    rows, columns = np.where(1 - selected_label)
    try:
        indices = np.random.randint(rows.shape[0], size=(num_points,))
        return np.stack([columns[indices], rows[indices]], axis=0).transpose()
    except:
        raise ValueError("The image slice is empty!")    
    

def generate_grid_points(label: np.ndarray, grid_distance: int) -> np.ndarray:
    """
    Generate grid points from `label`.
    """
    h, w = label.shape[0], label.shape[1]
    if grid_distance == 1:
        horizontal = np.arange(1, w - 1)
        vertical = np.arange(1, h - 1)
    else:
        horizontal = np.arange(grid_distance // 2, w - grid_distance // 2, grid_distance)
        vertical = np.arange(grid_distance // 2, h - grid_distance // 2, grid_distance)
    return np.transpose(np.stack(np.meshgrid(vertical, horizontal)), [1, 2, 0]).reshape(-1, 2)


def load_data(image_path: str) -> np.ndarray:
    """
    Load file `image_path` as a 3D image.
    """
    image_obj = nib.load(image_path)
    image_data: np.ndarray = image_obj.get_fdata()
    # Scale value to range [0, 256)
    min_, max_ = image_data.min(), image_data.max()
    image_data = (image_data - min_) / (max_ - min_) * 256
    return image_data  # dtype = np.float64

def gray_to_rgb(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to an RGB image.
    """
    image = np.expand_dims(grayscale_image, axis=2)
    image = np.repeat(image, 3, axis=2)
    return image.astype(np.uint8)

def get_slice(image: np.ndarray, slice_: int) -> np.ndarray:
    """
    Return a given z-slice of a 3D image.
    """
    return image[:, :, slice_]

"""
Helper functions that plot masks, points and boxes, from official code of SAM.
"""
def show_mask(mask: np.ndarray, ax: plt.Axes, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax: plt.Axes, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax: plt.Axes):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

"""
Load pretrained SAM model and build SAM predictor, from official code of SAM.
"""
def get_sam(model_type: str = 'vit_h',
            sam_checkpoint: str = '../sam_vit_h_4b8939.pth',
            device: str = 'cuda:0'):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return sam

def get_sam_predictor(model_type: str = 'vit_h',
                      sam_checkpoint: str = '../sam_vit_h_4b8939.pth',
                      device: str = 'cuda:0'):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamPredictor(sam)

def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_, truth_ = pred.astype(bool), truth.astype(bool)
    enumerator = (pred_ & truth_).sum() * 2
    denominator = pred_.sum() + truth_.sum()
    return enumerator / denominator

def compute_mdice(dice_dict):
    count = 0
    sum = 0
    for val in dice_dict.values():
        if not math.isnan(val):
            count += 1
            sum += val
    return sum / count



