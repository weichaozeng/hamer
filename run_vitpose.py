"""
Improved HaMeR pipeline with 3-pass architecture using Detectron2 + ViTPose + PoseTrack:
Pass 1: Extract all raw bboxes using Cascade Mask R-CNN & ViTPose
Pass 2: Clean bbox sequences globally 
Pass 3: Run HaMeR on cleaned bboxes
"""

from pathlib import Path
import torch
import argparse
import os
import sys
import cv2
import numpy as np
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import time
import random
import imageio
import yaml

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

# Detectron2 & ViTPose imports
import hamer
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
from vitpose_model import ViTPoseModel

# PoseTracker imports
from posetrack.posetracker import PoseTracker
from posetrack.wrapper import run_pose_tracker_wrapper
from posetrack.pa_nms import pa_nms


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices


def compute_bbox_from_keypoints(keypoints, img_shape, scale_factor=1.0, max_size_ratio=0.4):
    """Compute tight bbox from keypoints. Scaling is applied later when creating HaMeR input patches."""
    valid = keypoints[:, 2] > 0.5
    if valid.sum() < 10:
        return None
    
    valid_kp = keypoints[valid, :2]
    x_min, y_min = valid_kp.min(axis=0)
    x_max, y_max = valid_kp.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    width *= scale_factor
    height *= scale_factor
    
    img_h, img_w = img_shape[:2]
    max_width = img_w * max_size_ratio
    max_height = img_h * max_size_ratio
    
    width = min(width, max_width)
    height = min(height, max_height)
    size = max(width, height)
    
    x_min = cx - size / 2
    y_min = cy - size / 2
    x_max = cx + size / 2
    y_max = cy + size / 2
    
    return np.array([x_min, y_min, x_max, y_max])


def create_video_from_images(image_folder, output_video_path, fps=30):
    """Create MP4 video from images in a folder using imageio (same as src_cam video)."""
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"Warning: {image_folder} does not exist, skipping video creation")
        return
    
    # Get all jpg images sorted by filename
    image_files = sorted(list(image_folder.glob('*.jpg')))
    if len(image_files) == 0:
        print(f"Warning: No images found in {image_folder}, skipping video creation")
        return
    
    print(f"Creating video: {output_video_path}")
    
    # Load all images
    frames = []
    for img_file in tqdm(image_files, desc=f"  Loading frames"):
        img = imageio.imread(str(img_file))
        if img is not None:
            frames.append(img)
    
    if len(frames) == 0:
        print(f"Warning: Could not load any images from {image_folder}")
        return
    
    # Write video using imageio (same method as src_cam video generation)
    imageio.mimwrite(str(output_video_path), frames, fps=fps)
    print(f"  Video saved: {output_video_path} ({len(frames)} frames @ {fps} fps)")


# Helper function to expand bbox by a given scale (e.g., 1.2x) and make it SQUARE
def enlarge_bboxes(bboxes, scale=1.2, img_shape=None):
    if len(bboxes) == 0:
        return bboxes
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    
    # Make it square by taking the max dimension
    size = np.maximum(w, h) * scale
    
    new_x1 = cx - size / 2.0
    new_y1 = cy - size / 2.0
    new_x2 = cx + size / 2.0
    new_y2 = cy + size / 2.0
    
    # Optionally clip to image boundaries
    if img_shape is not None:
        H, W = img_shape[:2]
        new_x1 = np.clip(new_x1, 0, W - 1)
        new_y1 = np.clip(new_y1, 0, H - 1)
        new_x2 = np.clip(new_x2, 0, W - 1)
        new_y2 = np.clip(new_y2, 0, H - 1)
    if bboxes.shape[1] > 4:
        conf = bboxes[:, 4:5]
        new_bboxes = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
        return np.concatenate([new_bboxes, conf], axis=1)
    else:
        return np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)


def draw_bbox(img_cv2, id, box, is_right):
    x1, y1, x2, y2, conf = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color = (0, 0, 255) if is_right > 0 else (255, 0, 0)
    text = f'ID: {int(id)}'
    FONT_SCALE = 0.8
    THICKNESS_TEXT = 2
    TEXT_COLOR = (255, 255, 255)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS_TEXT)
    pt1 = (x1, y1)
    pt2 = (x1 + text_w, y1 + text_h + baseline)
    text_org = (x1, y1 + baseline + baseline)
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img_cv2, pt1, pt2, color, -1)
    cv2.putText(img_cv2, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS_TEXT)
    return img_cv2

BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Mid
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]
FINGER_COLORS = [
    (0, 0, 255),    # Thumb - Red
    (255, 0, 0),    # Index - Blue
    (0, 255, 0),    # Mid - Green
    (0, 255, 255),  # Ring - Yellow
    (255, 0, 255)   # Pinky - magenta
]
COLOR_KEYPOINT = (255, 255, 255) # Joint - White
COLOR_WRIST = (255, 165, 0)      # Wrist - Orange

def get_finger_color(bone_index: int) -> tuple:
    # Thumb (0-3) -> 0
    if bone_index < 4:
        return FINGER_COLORS[0]
    # Index (4-7) -> 1
    elif bone_index < 8:
        return FINGER_COLORS[1]
    # Mid (8-11) -> 2
    elif bone_index < 12:
        return FINGER_COLORS[2]
    # Ring (12-15) -> 3
    elif bone_index < 16:
        return FINGER_COLORS[3]
    # Pinky (16-19) -> 4
    else:
        return FINGER_COLORS[4]


def draw_pose(img_cv2, pose, thresh=0.5, K=21):
    if pose.shape != (K, 3):
        raise ValueError(f"Pose shape must be ({K}, 3), but got {pose.shape}")
    if isinstance(pose, torch.Tensor):
        keypoints = pose.cpu().numpy()
    else:
        keypoints = pose
    for i, (start_idx, end_idx) in enumerate(BONE_CONNECTIONS):
        kp_start = keypoints[start_idx]
        kp_end = keypoints[end_idx]
        if kp_start[2] > thresh and kp_end[2] > thresh:
            pt1 = (int(kp_start[0]), int(kp_start[1]))
            pt2 = (int(kp_end[0]), int(kp_end[1]))
            color = get_finger_color(i)
            cv2.line(img_cv2, pt1, pt2, color, 3)
    for i in range(K):
        kp = keypoints[i]
        if kp[2] > thresh:
            center = (int(kp[0]), int(kp[1]))
            if i == 0:
                color = COLOR_WRIST 
                radius = 6
            else:
                color = COLOR_KEYPOINT
                radius = 4
            cv2.circle(img_cv2, center, radius, color, -1) 
    return img_cv2


# ============================================================================
# PASS 1: Extract all raw bboxes using Detectron2 + ViTPose
# ============================================================================
def extract_raw_bboxes(img_paths, detector, cpm, tracker, tracker_args, vis_dir=None, det_thresh=0.5):
    """
    Pass 1: Extract raw hand bboxes using Detectron2 + ViTPose + PoseTracker.
    """
    print("\n" + "="*80)
    print("PASS 1: Extracting raw hand bboxes (Detectron2 + ViTPose + PoseTracker)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        os.makedirs(pass1_dir, exist_ok=True)
    
    raw_data_frame = []
    raw_data_track = {}

    for frame_idx, frame_path in enumerate(tqdm(sorted(img_paths), desc="Pass 1: Extracting hand bboxes")):
        frame_path = str(frame_path)
        frame_cv2 = cv2.imread(frame_path)
        img_rgb = frame_cv2[:, :, ::-1].copy()
        
        frame_data = {
            'has_det': False,
            'frame_idx': frame_idx,
            'frame_path': frame_path,
            'track_ids': None,
            'boxes': None, # will be [[x1, y1, x2, y2, conf_box]]
            'poses': None, # will be [[[x1, y1, conf_kp_1], ..., [x21, y21, conf_kp_21]]]
            'handedness': None,
        }
        
        bboxes_xyxy = []
        scores = []
        clses = []
        kps_xy = []
        kps_conf = []

        with torch.no_grad():
            with autocast():
                # 1. Detect Humans via Detectron2
                det_out = detector(frame_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > det_thresh)
                pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                pred_scores = det_instances.scores[valid_idx].cpu().numpy()

                if len(pred_bboxes) > 0:
                    # 2. Extract Human Keypoints via ViTPose
                    vitposes_out = cpm.predict_pose(
                        img_rgb,
                        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
                    )
                    
                    # Collect all left/right hands
                    for vitposes in vitposes_out:
                        left_hand_keyp = vitposes['keypoints'][-42:-21]  # 21 pts
                        right_hand_keyp = vitposes['keypoints'][-21:]    # 21 pts
                        
                        # Check left hand
                        l_valid = left_hand_keyp[:, 2] > 0.5
                        if sum(l_valid) > 3:
                            xmin, ymin = left_hand_keyp[l_valid, 0].min(), left_hand_keyp[l_valid, 1].min()
                            xmax, ymax = left_hand_keyp[l_valid, 0].max(), left_hand_keyp[l_valid, 1].max()
                            
                            bboxes_xyxy.append([xmin, ymin, xmax, ymax])
                            scores.append(left_hand_keyp[l_valid, 2].mean())
                            clses.append(0)  # 0: Left
                            kps_xy.append(left_hand_keyp[:, :2])
                            kps_conf.append(left_hand_keyp[:, 2])
                            
                        # Check right hand
                        r_valid = right_hand_keyp[:, 2] > 0.5
                        if sum(r_valid) > 3:
                            xmin, ymin = right_hand_keyp[r_valid, 0].min(), right_hand_keyp[r_valid, 1].min()
                            xmax, ymax = right_hand_keyp[r_valid, 0].max(), right_hand_keyp[r_valid, 1].max()
                            
                            bboxes_xyxy.append([xmin, ymin, xmax, ymax])
                            scores.append(right_hand_keyp[r_valid, 2].mean())
                            clses.append(1)  # 1: Right
                            kps_xy.append(right_hand_keyp[:, :2])
                            kps_conf.append(right_hand_keyp[:, 2])

                if len(bboxes_xyxy) > 0:
                    # 3. Pose NMS filtering
                    t_boxes = torch.tensor(bboxes_xyxy, dtype=torch.float32, device=device)
                    t_scores = torch.tensor(scores, dtype=torch.float32, device=device)
                    t_poses = torch.tensor(np.array(kps_xy).reshape(-1, 42), dtype=torch.float32, device=device)
                    t_pose_scores = torch.tensor(kps_conf, dtype=torch.float32, device=device)
                    nms_iou_thresh = getattr(tracker_args, 'nms_iou_thresh', 0.45)

                    keep_indices = pa_nms(
                        boxes=t_boxes,
                        scores=t_scores,
                        iou_threshold=nms_iou_thresh,
                        poses=t_poses,
                        pose_scores=t_pose_scores,
                        point_threshold=tracker_args.kp_thresh,
                        bone_threshold=tracker_args.bone_thresh
                    )
                    
                    keep_indices = keep_indices.cpu().numpy()
                    bboxes_xyxy = [bboxes_xyxy[i] for i in keep_indices]
                    scores = [scores[i] for i in keep_indices]
                    clses = [clses[i] for i in keep_indices]
                    kps_xy = [kps_xy[i] for i in keep_indices]
                    kps_conf = [kps_conf[i] for i in keep_indices]

                # 4. Run Tracker Wrapper
                tracked_results = run_pose_tracker_wrapper(
                    tracker, frame_cv2, bboxes_xyxy, scores, clses, kps_xy, kps_conf
                )

        frame_shape = frame_cv2.shape
        if len(tracked_results) > 0:
            track_ids = []
            final_boxes = []
            final_box_confs = []
            final_handedness = []
            final_poses = []
            final_pose_confs = []

            for res_dict in tracked_results:
                tid = res_dict['track_id']
                handedness = res_dict['handedness']
                cx, cy, w, h = res_dict['bbox_cxcywh']
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                
                # Assign default conf = 1.0 (since it survived NMS and tracking filters)
                conf = 1.0 
                
                # HaMeR uses OpenPose native format natively, so we just use the flattened [21,2] array directly.
                raw_kps_21x2 = res_dict['kps_flattened'].reshape(21, 2)
                conf_arr = np.where((raw_kps_21x2[:, 0] > 0) & (raw_kps_21x2[:, 1] > 0), 1.0, 0.0)
                
                track_ids.append(tid)
                final_boxes.append([x1, y1, x2, y2])
                final_box_confs.append(conf)
                final_handedness.append(handedness)
                final_poses.append(raw_kps_21x2)
                final_pose_confs.append(conf_arr)

            # Compose formatted numpy arrays to match Original Output Spec
            boxes_np = np.array(final_boxes)
            confs_np = np.array(final_box_confs)[:, None]
            boxes_with_conf = np.concatenate([boxes_np, confs_np], axis=1)
            
            enlarged_boxes = enlarge_bboxes(boxes_with_conf, scale=1.2, img_shape=frame_shape)
            
            frame_data['has_det'] = True
            frame_data['track_ids'] = np.array(track_ids)
            frame_data['boxes'] = enlarged_boxes
            frame_data['poses'] = np.concatenate([np.array(final_poses), np.array(final_pose_confs)[..., None]], axis=2)
            frame_data['handedness'] = np.array(final_handedness)

            # Maintain the Track Dictionary Mapping
            for i, track_id in enumerate(track_ids):
                if track_id not in raw_data_track:
                    raw_data_track[track_id] = {
                        'track_id': track_id,
                        'frame_indices': [],
                        'boxes': [],
                        'poses': [],
                        'handedness': [],
                    }
                raw_data_track[track_id]['frame_indices'].append(frame_idx)
                raw_data_track[track_id]['boxes'].append(enlarged_boxes[i])
                raw_data_track[track_id]['poses'].append(frame_data['poses'][i])
                raw_data_track[track_id]['handedness'].append(final_handedness[i])
        
        # Visualize raw bboxes
        if vis_dir is not None:
            frame_vis = frame_cv2.copy()
            if frame_data['has_det']:
                for i, track_id in enumerate(frame_data['track_ids']):
                    frame_vis = draw_bbox(frame_vis, track_id, frame_data['boxes'][i], frame_data['handedness'][i])
                    frame_vis = draw_pose(frame_vis, frame_data['poses'][i])
            img_fn = os.path.splitext(os.path.basename(frame_path))[0]
            cv2.imwrite(os.path.join(pass1_dir, f'{img_fn}_raw.jpg'), frame_vis)
        
        raw_data_frame.append(frame_data)
    
    # Create video from Pass 1 visualizations
    if vis_dir is not None:
        pass1_dir = os.path.join(vis_dir, 'pass1_raw_bboxes')
        video_path = os.path.join(vis_dir, 'pass1_raw_bboxes.mp4')
        create_video_from_images(pass1_dir, video_path, fps=30)
    
    return raw_data_frame, raw_data_track


# ============================================================================
# PASS 2: Clean bbox sequences globally
# ============================================================================
def clean_bbox_sequences(raw_data_frame, raw_data_track, interpolate=5, vis_dir=None):
    """
    Pass 2: Clean bbox sequences using global temporal information.
    """
    print("\n" + "="*80)
    print("PASS 2: Cleaning bbox sequences")
    print("="*80)
    
    if vis_dir is not None:
        pass2_dir = os.path.join(vis_dir, 'pass2_cleaned_bboxes')
        os.makedirs(pass2_dir, exist_ok=True)
    
    # Plot trajectories BEFORE cleaning
    if vis_dir is not None:
        plot_bbox_trajectories(raw_data_track, vis_dir, filename='pass2_bbox_trajectories_before_cleaning.png')
    
    cleaned_data_frame, cleaned_data_track = fix_handedness_inconsistencies(raw_data_frame, raw_data_track)

    if interpolate > 0:
        cleaned_data_frame, cleaned_data_track = interpolate_missing_bboxes(cleaned_data_frame, cleaned_data_track, max_gap=interpolate)

    # Visualize before/after comparison
    if vis_dir is not None:
        for i, frame_data in enumerate(cleaned_data_frame):
            frame_cv2 = cv2.imread(frame_data['frame_path'])
            frame_vis = frame_cv2.copy()
            if frame_data['has_det']:
                for j, track_id in enumerate(frame_data['track_ids']):
                    frame_vis = draw_bbox(frame_vis, track_id, frame_data['boxes'][j], frame_data['handedness'][j])
                    frame_vis = draw_pose(frame_vis, frame_data['poses'][j])
            img_fn = os.path.splitext(os.path.basename(frame_data['frame_path']))[0]
            cv2.imwrite(os.path.join(pass2_dir, f'{img_fn}_cleaned.jpg'), frame_vis)
    
    # Create video from Pass 2 visualizations
    if vis_dir is not None:
        pass2_dir = os.path.join(vis_dir, 'pass2_cleaned_bboxes')
        video_path = os.path.join(vis_dir, 'pass2_cleaned_bboxes.mp4')
        create_video_from_images(pass2_dir, video_path, fps=30)
        plot_bbox_trajectories(cleaned_data_track, vis_dir, filename='pass2_bbox_trajectories_after_cleaning.png')
    
    return cleaned_data_frame, cleaned_data_track


def fix_handedness_inconsistencies(raw_data_frame, raw_data_track):
    """Fix handedness inconsistencies for each track by voting and update to data frame."""
    track_id_to_fixed_handedness = {}
    for track_id, track_data in raw_data_track.items():
        handedness_list = track_data['handedness']
        counts = Counter([int(h) for h in handedness_list])
        dominant_handedness = counts.most_common(1)[0][0]
        track_id_to_fixed_handedness[track_id] = dominant_handedness
        raw_data_track[track_id]['handedness'] = [float(dominant_handedness)] * len(handedness_list)
    for frame_data in raw_data_frame:
        if not frame_data['has_det']:
            continue
        current_track_ids = frame_data['track_ids']
        current_handedness = frame_data['handedness']
        for i, trk_id in enumerate(current_track_ids):
            correct_h = track_id_to_fixed_handedness[trk_id]
            current_handedness[i] = float(correct_h)
        frame_data['handedness'] = current_handedness
    return raw_data_frame, raw_data_track


def interpolate_missing_bboxes(cleaned_data_frame, cleaned_data_track, max_gap=5):
    """Interpolate missing bboxes for short gaps and create dummy keypoints."""
    total_interpolated = 0
    for track_id, track_data in cleaned_data_track.items():
        frame_indices = track_data['frame_indices']
        boxes = track_data['boxes']          
        poses = track_data['poses']          
        handedness = track_data['handedness']
        if len(frame_indices) < 2:
            continue
        
        new_frame_indices = []
        new_boxes = []
        new_poses = []
        new_handedness = []
        
        for i in range(len(frame_indices) - 1):
            curr_f = frame_indices[i]
            next_f = frame_indices[i+1]
            
            new_frame_indices.append(curr_f)
            new_boxes.append(boxes[i])
            new_poses.append(poses[i])
            new_handedness.append(handedness[i])
            
            delta = next_f - curr_f
            if 1 < delta <= (max_gap + 1):
                start_box = boxes[i]        # shape [5] (x1,y1,x2,y2,conf)
                end_box = boxes[i+1]
                start_pose = poses[i]       # shape [21, 3]
                end_pose = poses[i+1]
                curr_hand_cls = handedness[i] 
                for gap_i in range(1, delta):
                    interp_f = curr_f + gap_i
                    alpha = gap_i / delta
                    interp_box = (1 - alpha) * start_box + alpha * end_box
                    interp_pose = (1 - alpha) * start_pose + alpha * end_pose
                    new_frame_indices.append(interp_f)
                    new_boxes.append(interp_box)
                    new_poses.append(interp_pose)
                    new_handedness.append(curr_hand_cls)
                    total_interpolated += 1

                    frame_data = cleaned_data_frame[interp_f]
                    box_to_add = interp_box[None, :]        # [1, 5]
                    pose_to_add = interp_pose[None, :, :]   # [1, 21, 3]
                    id_to_add = np.array([track_id])        # [1]
                    hand_to_add = np.array([curr_hand_cls]) # [1]
                    
                    if not frame_data['has_det']:
                        frame_data['has_det'] = True
                        frame_data['track_ids'] = id_to_add
                        frame_data['boxes'] = box_to_add
                        frame_data['poses'] = pose_to_add
                        frame_data['handedness'] = hand_to_add
                    else:
                        frame_data['track_ids'] = np.concatenate([frame_data['track_ids'], id_to_add])
                        frame_data['boxes'] = np.concatenate([frame_data['boxes'], box_to_add], axis=0)
                        frame_data['poses'] = np.concatenate([frame_data['poses'], pose_to_add], axis=0)
                        frame_data['handedness'] = np.concatenate([frame_data['handedness'], hand_to_add])
        
        last_idx = len(frame_indices) - 1
        new_frame_indices.append(frame_indices[last_idx])
        new_boxes.append(boxes[last_idx])
        new_poses.append(poses[last_idx])
        new_handedness.append(handedness[last_idx])
        
        cleaned_data_track[track_id]['frame_indices'] = new_frame_indices
        cleaned_data_track[track_id]['boxes'] = new_boxes
        cleaned_data_track[track_id]['poses'] = new_poses
        cleaned_data_track[track_id]['handedness'] = new_handedness

    print(f"Interpolation complete. Filled {total_interpolated} missing detections.")
    return cleaned_data_frame, cleaned_data_track
    

def plot_bbox_trajectories(data_track, vis_dir, filename='bbox_trajectories.png', wrist=False):
    """
    Plot hand bbox center / wrist trajectories over time.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8), dpi=120)
    ax = plt.gca()
    cmap_left = plt.cm.Blues
    cmap_right = plt.cm.Reds
   
    for track_id, data in data_track.items():
        boxes = np.array(data['boxes']) # Shape: [N, 5]
        poses = np.array(data['poses']) # Shape: [N, 21, 3]
        handedness = data['handedness']
        frame_indices = data['frame_indices']
        if not wrist:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
        else: 
            cx = poses[:, 0]
            cy = poses[:, 1]

        is_right = int(handedness[0]) == 1
        hand_label = "Right" if is_right else "Left"
        
        random.seed(track_id)
        color_intensity = 0.5 + (random.random() * 0.4)

        if is_right:
            color = cmap_right(color_intensity)
            marker_style = 'o'
        else:
            color = cmap_left(color_intensity)
            marker_style = '^'
        
        gap_indices = np.where(np.diff(frame_indices) > 1)[0]
        def plot_segment(x_arr, y_arr, style, alpha_val):
            ax.plot(x_arr, y_arr, color=color, linestyle=style, 
                    linewidth=2, alpha=alpha_val)

        start_idx = 0
        for gap_idx in gap_indices:
            if gap_idx > start_idx:
                plot_segment(cx[start_idx : gap_idx + 1], cy[start_idx : gap_idx + 1], '-', 0.8)
            plot_segment(cx[gap_idx : gap_idx + 2], cy[gap_idx : gap_idx + 2], '--', 0.4)
            start_idx = gap_idx + 1
        if start_idx < len(frame_indices):
            plot_segment(cx[start_idx:], cy[start_idx:], '-', 0.8)

        if len(cx) > 0:
            ax.scatter(cx[0], cy[0], color=color, s=60, marker=marker_style, edgecolors='k', zorder=5)
            ax.scatter(cx[-1], cy[-1], color=color, s=60, marker='x', linewidths=2.5, zorder=5)
            ax.text(cx[0], cy[0], f'{track_id}', fontsize=8, 
                    color='white', fontweight='bold', 
                    bbox=dict(facecolor=color, alpha=0.9, edgecolor='none', pad=1),
                    zorder=6)
            
    ax.set_title("Hand Trajectories (Solid=Continuous, Dashed=Gap/Interpolated)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis() 
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=cmap_left(0.7), lw=2, label='Left Hand'),
        Line2D([0], [0], color=cmap_right(0.7), lw=2, label='Right Hand'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Continuous'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Gap/Jump')
    ]
    ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.2, 1))
    
    plt.tight_layout()
    save_path = os.path.join(vis_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PASS 3: Run HaMeR on cleaned bboxes
# ============================================================================

def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    """Convert cropped coordinates to original image coordinates."""
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    keypoints *= h[..., None, None] / crop_size
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints


def run_hamer_on_cleaned_bboxes(cleaned_data_frame, model, model_cfg, renderer, args, dummy_keypoints=True):
    """
    Pass 3: Run HaMeR on cleaned bboxes.
    """
    print("\n" + "="*80)
    print("PASS 3: Running HaMeR on cleaned bboxes")
    print("="*80)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    result_data_frame = {}
    result_data_track = {}

    for frame_data in tqdm(cleaned_data_frame, desc="Pass 3: Running HaMeR"):
        frame_path = frame_data['frame_path']
        frame_idx = frame_data['frame_idx']
        img_cv2 = cv2.imread(frame_path)
        img_fn = os.path.splitext(os.path.basename(frame_path))[0]
        
        if not frame_data['has_det']:
            result_data_frame[frame_path] = {
                'boxes': [],
                'pred_cam': [],
                'mano': [],
                'cam_trans': [],
                'tracked_ids': [],
                'handedness': [],
                'extra_data': [],
                'verts': [],
                'shot': 0
            }
            if args.render:
                render_path = os.path.join(os.path.dirname(args.res_folder), f'render_all')
                os.makedirs(render_path, exist_ok=True)
                cv2.imwrite(os.path.join(render_path, f'{img_fn}.jpg'), img_cv2)
            continue
        
        track_ids = frame_data['track_ids']
        bboxes = frame_data['boxes'][:, :4]  # Exclude confidence
        is_right = frame_data['handedness']
        keypoints = frame_data['poses']

        # Create dataset
        dataset = ViTDetDataset(model_cfg, img_cv2, bboxes, is_right, keypoints, track_ids, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Run HaMeR
        all_pred_cam = []
        all_verts = []
        all_cam_t = []
        all_right = []
        all_mano_params = []
        all_pred_2d = []
        all_bboxes = []
        all_poses = []
        all_track_ids = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            
            # Compute camera translation (SLAHMR-style)
            batch_size = batch['img'].shape[0]
            pred_cam_t_full = torch.zeros(batch_size, 3, device=pred_cam.device)
            
            for n in range(batch_size):
                cam = pred_cam[n]
                H, W = img_size[n, 1], img_size[n, 0]
                focal = scaled_focal_length[n] if scaled_focal_length.ndim > 0 else scaled_focal_length
                cx, cy = box_center[n]
                scale = box_size[n]
                
                tz = 2 * focal / (scale * cam[0] + 1e-6)
                tx = cam[1] + tz / focal * (cx - W / 2)
                ty = cam[2] + tz / focal * (cy - H / 2)
                
                pred_cam_t_full[n] = torch.tensor([tx, ty, tz], device=pred_cam.device)
            
            pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()
            
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                pred_joints = out['pred_keypoints_2d'][n].detach().cpu().numpy()
                is_right_val = int(batch['right'][n].cpu().numpy())
                verts[:, 0] = (2 * is_right_val - 1) * verts[:, 0]
                pred_joints[:, 0] = (2 * is_right_val - 1) * pred_joints[:, 0]
                cam_t = pred_cam_t_full[n]
                track_id = batch['track_id'][n].detach().cpu().numpy().item()
                mano_params = out['pred_mano_params'][n]
                mano_params['is_right'] = is_right_val

                all_pred_cam.append(pred_cam[n].detach().cpu().numpy())
                all_mano_params.append(mano_params)
                all_cam_t.append(cam_t)
                all_right.append(is_right_val)
                all_pred_2d.append(pred_joints)
                all_bboxes.append(batch['bbox'][n].detach().cpu().numpy())
                all_poses.append(batch['2d'][n].detach().cpu().numpy())
                all_track_ids.append(track_id)
                all_verts.append(verts)

                if track_id not in result_data_track:
                    result_data_track[track_id] = {
                        'tracked_id': track_id,
                        'frame_indices': [],
                        'boxes': [],
                        'poses': [],
                        'pred_cam': [],
                        'mano': [],
                        'cam_trans': [],
                        'handedness': [],
                        'extra_data': [],
                        'verts': [],
                        'shot': 0
                    }
                result_data_track[track_id]['frame_indices'].append(frame_idx)
                result_data_track[track_id]['boxes'].append(batch['bbox'][n].detach().cpu().numpy())
                result_data_track[track_id]['poses'].append(batch['2d'][n].detach().cpu().numpy())
                result_data_track[track_id]['pred_cam'].append(pred_cam[n].detach().cpu().numpy())
                result_data_track[track_id]['mano'].append(mano_params)
                result_data_track[track_id]['cam_trans'].append(cam_t)
                result_data_track[track_id]['handedness'].append(is_right_val)
                result_data_track[track_id]['extra_data'].append(pred_joints)
                result_data_track[track_id]['verts'].append(verts)  

        
        # Convert 2D keypoints to original image coordinates
        if len(all_pred_2d) > 0:
            all_pred_2d_np = np.stack(all_pred_2d)
            all_bboxes_np = np.stack(all_bboxes)
            
            # Add confidence column
            v = np.ones((all_pred_2d_np.shape[0], all_pred_2d_np.shape[1], 1))
            all_pred_2d_np = np.concatenate((all_pred_2d_np, v), axis=-1)
            
            # Convert from crop coords to original image coords
            all_pred_2d_np = model_cfg.MODEL.IMAGE_SIZE * (all_pred_2d_np + 0.5)
            all_pred_2d_np = convert_crop_coords_to_orig_img(bbox=all_bboxes_np, keypoints=all_pred_2d_np, crop_size=model_cfg.MODEL.IMAGE_SIZE)
            all_pred_2d_np[:, :, -1] = 1  # Set all confidences to 1
            
            extra_data = [all_pred_2d_np[i].tolist() for i in range(len(all_pred_2d_np))]
        else:
            extra_data = []
        
        # Store results
        result_data_frame[frame_path] = {
            'boxes': all_bboxes,
            'pred_cam': all_pred_cam,
            'mano': all_mano_params,
            'cam_trans': all_cam_t,
            'handedness': all_right,
            'tracked_ids': all_track_ids,
            'extra_data': extra_data,
            'verts': all_verts,
            'shot': 0
        }
        
        # Render hand meshes if requested
        if args.render and len(all_verts) > 0 and renderer is not None:
            LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            
            cam_view, _ = renderer.render_rgba_multiple(
                all_verts, 
                cam_t=all_cam_t, 
                render_res=img_size[0], 
                is_right=all_right, 
                **misc_args
            )
            
            # Overlay on original image
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            
            # Save rendered result
            render_path = os.path.join(os.path.dirname(args.res_folder), f'render_all')
            os.makedirs(render_path, exist_ok=True)
            cv2.imwrite(os.path.join(render_path, f'{img_fn}.jpg'), 255 * input_img_overlay[:, :, ::-1])
    
    # Create video from Pass 3 rendered results
    render_path = os.path.join(os.path.dirname(args.res_folder), f'render_all')
    if os.path.exists(render_path):
        seq_name = os.path.basename(os.path.dirname(os.path.dirname(args.res_folder)))
        video_path = os.path.join(os.path.dirname(args.res_folder), f'{seq_name}_render_all.mp4')
        create_video_from_images(render_path, video_path, fps=30)
    
    return result_data_frame, result_data_track


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HaMeR Pipeline with Detectron2 + ViTPose + PoseTracker')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--res_folder', type=str, help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=1.3, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')
    parser.add_argument('--conf', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--render', dest='render', action='store_true', default=False, help='If set, render video/vis results')
    
    # New Tracker & Detector arguments
    parser.add_argument('--det_thresh', type=float, default=0.4, help='Detectron2 Confidence Threshold')
    parser.add_argument("--tracker_cfg", type=str, default="/home/zvc/Project/hamer/posetrack/posetrack.yaml", help="Path to tracker YAML config file")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_folder = Path(args.img_folder)
    
    # Get all images matching file_type patterns
    img_paths = sorted([img for end in args.file_type for img in img_folder.glob(end)])
    print(f"Found {len(img_paths)} images in {img_folder}")
    
    # Load HaMeR
    CACHE_DIR_HAMER = args.checkpoint
    print(f"\nLoading HaMeR models from {CACHE_DIR_HAMER}/_DATA/hamer_ckpts")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # ---------------------------------------------------------
    # NEW: Load Detectron2, ViTPose, and PoseTracker Config
    # ---------------------------------------------------------
    print("\nLoading Detectron2 Human Detector...")
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    print("\nLoading ViTPose Model...")
    cpm = ViTPoseModel(device)

    print(f"\nLoading Tracker Config from {args.tracker_cfg}...")
    with open(args.tracker_cfg, 'r') as f:
        tracker_cfg_dict = yaml.safe_load(f)
    tracker_args = argparse.Namespace(**tracker_cfg_dict)
    
    # Initialize PoseTracker once per video sequence 
    tracker = PoseTracker(tracker_args, frame_rate=30)
    # ---------------------------------------------------------

    # Load renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    # Create visualization directory
    vis_dir = None
    if args.render and args.res_folder is not None:
        vis_dir = os.path.join(os.path.dirname(args.res_folder), f'bbox_vis')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"\nBbox visualizations will be saved to: {vis_dir}")
    
    # PASS 1: Extract raw bboxes using Detectron2+ViTPose+PoseTracker
    raw_data_frame, raw_data_track = extract_raw_bboxes(
        img_paths, 
        detector, 
        cpm, 
        tracker, 
        tracker_args, 
        vis_dir=vis_dir, 
        det_thresh=args.det_thresh
    )
    
    # PASS 2: Clean bbox sequences
    cleaned_data_frame, cleaned_data_track = clean_bbox_sequences(raw_data_frame, raw_data_track, vis_dir=vis_dir)
    
    # PASS 3: Run HaMeR
    result_data_frame, result_data_track = run_hamer_on_cleaned_bboxes(cleaned_data_frame, model, model_cfg, renderer, args)
    
    # Save results
    output_path = Path(args.res_folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result_data_frame, f)

    track_dir = os.path.join(os.path.dirname(args.res_folder), f'track')
    os.makedirs(track_dir, exist_ok=True)
    for track_id, track_data in result_data_track.items():
        track_path = os.path.join(track_dir, f'track_{track_id:03d}.pkl')
        with open(track_path, 'wb') as f:
            pickle.dump(track_data, f)
    
    print(f"\nResults saved to {output_path}")
    return 0

if __name__ == '__main__':
    main()