import cv2
import os
import time
import random
import math
import numpy as np
import tensorflow as tf
from data_util import GeneratorEnqueuer
from shapely.geometry import Polygon
import sys
from nets import utils
import imageio

# Read image names from split file
def get_set_list(data_path, split = 'train'):
	set_list = []
	split_file = os.path.join(data_path, split + '.txt')
	assert os.path.exists(split_file), "%s is not exists !\n" % split_file
	with open(split_file, 'r') as f:
		for line in f.readlines():
			set_list.append(line.strip())
	return set_list

# Read .bmp .jpg .gif
def read_image(file):
	im = cv2.imread(file)
	if im is None:
		gif = imageio.mimread(file)
		if gif is not None:
			gif = np.array(gif)
			gif = gif[0]
			im = gif[:, :, 0:3].astype(np.float32)
	else:
		im = im.astype(np.float32)
	return im

# Get image and image name 
def get_image(im_name, image_path):
	extension = ['.jpg', '.png', '.gif', '.bmp', '.jpeg']
	im_fn = None
	for ext in extension:
		im_fn = os.path.join(image_path, im_name + ext)
		if os.path.exists(im_fn):
			break
	if im_fn is not None:
		im = read_image(im_fn)
	return im, im_fn

# Compute area of a polygon
# poly np.array() shape:[4,2]
def polygon_area(poly):
	poly = poly.astype(np.float32)
	edge = [
	    (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
	    (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
	    (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
	    (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
			]
	return np.sum(edge)/2.

# Extract annotation from .txt and transform to bbox and masks
# return:
# 		mask  : [h, w, instance_num] dtype=np.uint8
def get_annotation(im_name, anno_path, im_shape):
	anno_fn = os.path.join(anno_path, im_name + '.txt')
	assert os.path.exists(anno_fn), \
			"%s dones't exists !" % anno_fn
	with open(anno_fn, 'r') as f:
		masks = []
		for line in f.readlines():
			split_line = line.strip().split(',')
			# Judge if str is hard samples 
			if len(split_line) >= 8 and not ('###' in line):
				poly = list(map(int, split_line[:8]))
				poly = np.array(poly).reshape((4,2)).astype(np.int32)
				poly_area = polygon_area(poly)
				# filter invalid polygon
				if abs(poly_area) < 1:
					continue
				# if the poly is too small, then ignore it during training
				poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
				poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
				if poly_h < 10 or poly_w < 10:
					continue
				# generate text mask for per text region poly
				text_mask = np.zeros((im_shape[0], im_shape[1]), dtype=np.uint8)
				cv2.fillPoly(text_mask, poly[np.newaxis, :, :], 1)
				masks.append(text_mask)
		masks = np.array(masks).reshape((-1, im_shape[0], im_shape[1])).astype(np.bool)
		masks = masks.transpose(1, 2, 0)
	return masks

def resize_image_and_annotation(image, mask, cfg):
	image, window, scale, padding, crop = utils.resize_image(
		image,
		min_dim=cfg.IMAGE_MIN_DIM,
		min_scale=cfg.IMAGE_MIN_SCALE,
		max_dim=cfg.IMAGE_MAX_DIM,
		mode=cfg.IMAGE_RESIZE_MODE)
	mask = utils.resize_mask(mask, scale, padding, crop)
	# random flip image
	if random.randint(0, 1):
		image = np.fliplr(image)
		mask = np.fliplr(mask)
    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
	_idx = np.sum(mask, axis=(0, 1)) > 0
	mask = mask[:, :, _idx]
	bbox = utils.extract_bboxes(mask)
	
	# gt class id: [background, text]
	class_id = np.ones((mask.shape[-1])).astype(np.int32)
	return image, bbox, mask, class_id


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

# Generater training data
def generator(data_path, cfg):
	batch_size = cfg.BATCH_SIZE
	train_list = get_set_list(data_path, split='train')
	print('{} training images in {}'.format(\
			len(train_list), data_path))
	image_path = os.path.join(data_path, 'JPEGImages')
	annotation_path = os.path.join(data_path, 'Annotations')
	assert os.path.exists(image_path), \
			"%s dones't exists, check it !" % image_path
	assert os.path.exists(annotation_path),\
			"%s dones't exists, check it !" % annotation_path
	# Anchors
	# [anchor_count, (y1, x1, y2, x2)]
	backbone_shapes = compute_backbone_shapes(cfg, cfg.IMAGE_SHAPE)
	anchors = utils.generate_pyramid_anchors(cfg.RPN_ANCHOR_SCALES,
											cfg.RPN_ANCHOR_RATIOS,
											backbone_shapes,
											cfg.BACKBONE_STRIDES,
											cfg.RPN_ANCHOR_STRIDE)
	b = 0  # batch item index
	while True:
		random.shuffle(train_list)
		for index in train_list:
			try:
				im, im_fn = get_image(index, image_path)
				if im_fn is None or im.shape[0] == 0 or im.shape[1] == 0 or im.shape[2] != 3:
					print(image_path, " %d image is empty !" % index)
					continue
				im_mask = get_annotation(index, annotation_path, im.shape)
				# Skip images that have no instances. This can happen in cases
	            # where we train on a subset of classes and the image doesn't
	            # have any of the classes we care about.
				if im_mask.shape[-1] == 0:
					continue
				# Original image and annotation resize to input shape
				image, gt_boxes, gt_masks, gt_class_ids = resize_image_and_annotation(im, im_mask, cfg)
				gt_global_mask = np.sum(gt_masks, axis=-1).reshape((image.shape[0], image.shape[1])).astype(np.uint8)
				# use mini_mask to reduce memory cost
				if cfg.USE_MINI_MASK:
					gt_masks = utils.minimize_mask(gt_boxes, gt_masks, cfg.MINI_MASK_SHAPE)
				# RPN Targets
				rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
														gt_class_ids, gt_boxes, cfg)
				# If more instances than fits in the array, sub-sample from them.
				if gt_boxes.shape[0] > cfg.MAX_GT_INSTANCES:
					ids = np.random.choice(
						np.arange(gt_boxes.shape[0]), cfg.MAX_GT_INSTANCES, replace=False)
					gt_class_ids = gt_class_ids[ids]
					gt_boxes = gt_boxes[ids]
					gt_masks = gt_masks[:, :, ids]
				# Init batch arrays
				if b == 0:
					batch_rpn_match = np.zeros(
						[batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
					batch_rpn_bbox = np.zeros(
						[batch_size, cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
					batch_images = np.zeros(
						(batch_size,) + image.shape, dtype=np.float32)
					batch_gt_class_ids = np.zeros(
						(batch_size, cfg.MAX_GT_INSTANCES), dtype=np.int32)
					batch_gt_boxes = np.zeros(
						(batch_size, cfg.MAX_GT_INSTANCES, 4), dtype=np.int32)
					batch_gt_masks = np.zeros(
						(batch_size, gt_masks.shape[0], gt_masks.shape[1],
					 	cfg.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
					batch_gt_global_masks = np.zeros(
						(batch_size,) + gt_global_mask.shape, dtype=gt_global_mask.dtype)
				# Add to batch
				batch_rpn_match[b] = rpn_match[:, np.newaxis]
				batch_rpn_bbox[b] = rpn_bbox
				batch_images[b] = image
				batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
				batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
				batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
				batch_gt_global_masks[b] = gt_global_mask
				b += 1
				if b >= batch_size:
					inputs = {}
					inputs['input_image'] = batch_images
					inputs['input_rpn_match'] = batch_rpn_match
					inputs['input_rpn_bbox'] = batch_rpn_bbox
					inputs['input_gt_class_ids'] = batch_gt_class_ids
					inputs['input_gt_boxes'] = batch_gt_boxes
					inputs['input_gt_masks'] = batch_gt_masks
					inputs['input_gt_global_masks'] = batch_gt_global_masks
					# get batch data
					yield inputs
					b = 0
			except Exception as e:
				import traceback
				traceback.print_exc()
				continue


def get_batch(num_workers, **kwargs):
	enqueuer = None
	try:
		enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
		print('Generator use 10 batches for buffering, this may take a while')
		enqueuer.start(max_queue_size = 10, workers = num_workers)
		generator_output = None
		while True:
			while enqueuer.is_running():
				if not enqueuer.queue.empty():
					generator_output = enqueuer.queue.get()
					break
				else:
					time.sleep(0.01)
			yield generator_output
			generator_output = None
	finally:
		if enqueuer is not None:
			enqueuer.stop()
'''
def debug():
	im = cv2.imread("data/debug/img_2.png")
	im_mask = get_annotation("gt_img_2", "data/debug", im.shape)
	global_mask = np.sum(im_mask, axis=-1).reshape((im.shape[0], im.shape[1]))
	global_mask = global_mask.astype(np.uint8)
	global_mask[global_mask > 0] = 255 
	cv2.imwrite("data/debug/global_mask_2.jpg", global_mask)
	print(global_mask.shape)
'''

if __name__ == "__main__":
	batch_data = get_batch(num_workers=10, data_path="data/train")
	inputs = next(batch_data)
	for key in inputs.keys():
		print("inputs[%s].shape = " % key, inputs[key].shape)