import sys
sys.path.append('..')
import utils
import numpy as np
from typing import List, Tuple, Optional, Union
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
import torch

def prompt_for_target(input_label: np.ndarray,
						target: int,
						point_prompt: List[str],
						bounding_box_prompt: bool,
						bounding_box_margin: int) \
	  -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
	cur_label = utils.select_label(input_label, target)
	if cur_label.sum() == 0:
		return None, None, None

	center_point = utils.find_center(cur_label)[None, :] if 'center' in point_prompt \
					else np.ones((0, 2), dtype=np.int64)
	center_label = np.ones((center_point.shape[0], ), dtype=np.int8)

	fg_points = utils.find_fg_random(cur_label, point_prompt.count('random'))
	fg_labels = np.ones((fg_points.shape[0], ), dtype=np.int8)

	bg_points = utils.find_bg_random(cur_label, point_prompt.count('bg_random'))
	bg_labels = np.zeros((bg_points.shape[0], ), dtype=np.int8)

	points = np.concatenate([center_point, fg_points, bg_points], axis=0)
	labels = np.concatenate([center_label, fg_labels, bg_labels], axis=0)

	if points.shape[0] == 0:
		points = None
		labels = None

	box = None
	if bounding_box_prompt:
		box = utils.find_bounding_box(cur_label, bounding_box_margin)

	return points, labels, box

def generate_prompt(input_label: np.ndarray,
					point_prompt: List[str],
					bounding_box_prompt: bool,
					bounding_box_margin: int,
					targets: Optional[Union[int, List[int]]] = list(range(1, 14))) \
      -> Tuple[List[int], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
	
	if targets is None:
		targets = list(range(1, 14))
	elif isinstance(targets, int):
		targets = [targets]

	sum_points, sum_labels, sum_box = [], [], []
	valid_targets = []
	for target in targets:
		points, labels, box = prompt_for_target(input_label, target, point_prompt, bounding_box_prompt, bounding_box_margin)
		if points is None and labels is None and box is None:
			continue
		valid_targets.append(target)
		if points is not None:
			sum_points.append(points)
		if labels is not None:
			sum_labels.append(labels)
		if box is not None:
			sum_box.append(box)

	if len(valid_targets) == 0:
		return [], None, None, None
	if len(sum_points) == 0:
		sum_points = None
	else:
		sum_points = np.stack(sum_points, axis=0)
	if len(sum_labels) == 0:
		sum_labels = None
	else:
		sum_labels = np.stack(sum_labels, axis=0)
	if len(sum_box) == 0:
		sum_box = None
	else:
		sum_box = np.stack(sum_box, axis=0)
	return valid_targets, sum_points, sum_labels, sum_box

def generate_prompt_input(sam: Sam,
							data: np.ndarray, 
							labels: np.ndarray,
							z_batch_range: slice,
							point_prompt: List[str],
							bounding_box_prompt: bool,
							bounding_box_margin: int,
							targets: Optional[Union[int, List[int]]] = None)\
					-> Tuple[List[Tuple[int, List[int]]], List[dict]]:
	
	transform = ResizeLongestSide(sam.image_encoder.img_size)

	target_list = []
	sam_input = []

	for i in z_batch_range:
		image_i = torch.from_numpy(transform.apply_image(data[..., i])).to(sam.device).permute(2, 0, 1).contiguous()

		gen_targets, point_coords, point_labels, box = generate_prompt(labels[..., i], point_prompt, bounding_box_prompt, bounding_box_margin, targets)

		if point_coords is None and point_labels is None and box is None:
			continue

		target_list.append((i, gen_targets))
		input_dict_i = {}
		input_dict_i['image'] = image_i
		input_dict_i['original_size'] = data.shape[:2]
		if point_coords is not None:
			point_coords = transform.apply_coords_torch(torch.from_numpy(point_coords).to(sam.device), data.shape[:2])
			input_dict_i['point_coords'] = point_coords
		if point_labels is not None:
			point_labels = torch.from_numpy(point_labels).to(sam.device)
			input_dict_i['point_labels'] = point_labels
		if box is not None:
			box = transform.apply_boxes_torch(torch.from_numpy(box).to(sam.device), data.shape[:2])
			input_dict_i['box'] = box
		sam_input.append(input_dict_i)

	return target_list, sam_input