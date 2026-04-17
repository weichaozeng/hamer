import torch
import torch.nn.functional as F

# 预先定义好骨骼连接 (20对)，并将其放到指定的 device 上运算
_BONE_CONNECTIONS_LIST = [
    [0,1], [1,2], [2,3], [3,4],
    [0,5], [5,6], [6,7], [7,8],
    [0,9], [9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
]

def pa_nms(
    boxes: torch.Tensor, 
    scores: torch.Tensor, 
    iou_threshold: float, 
    poses: torch.Tensor, 
    pose_scores: torch.Tensor, 
    point_threshold: float, 
    bone_threshold: float
) -> torch.Tensor:
    """Optimized Pose-Aware NMS with early termination.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
        scores (torch.Tensor): Confidence scores with shape (N,).
        iou_threshold (float): IoU threshold for suppression.
        poses (torch.Tensor): Keypoints with shape (N, 42) in xy flattened format.
        pose_scores (torch.Tensor): Confidence scores with shape (N, 21).
        point_threshold (float): Threshold for point similarity for suppression.
        bone_threshold (float): Threshold for bone similarity for suppression.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    device = boxes.device
    BONE_CONNECTIONS = torch.tensor(_BONE_CONNECTIONS_LIST, dtype=torch.long, device=device)
    
    # bbox
    x1, y1, x2, y2 = boxes.unbind(1)
    widths = (x2 - x1).clamp(min=1e-6)
    heights = (y2 - y1).clamp(min=1e-6)
    box_scales = torch.sqrt(widths**2 + heights**2).view(-1, 1, 1) # (N, 1, 1)
    areas = (x2 - x1) * (y2 - y1)

    # pose
    N = poses.shape[0]
    kps = poses.reshape(N, 21, 2)  # (N, 21, 2)
    parent_idx = BONE_CONNECTIONS[:, 0] # (20,)
    child_idx = BONE_CONNECTIONS[:, 1]  # (20,)
    node_valid = pose_scores > 0.5
    valid_nums = node_valid.sum(dim=1)

    # P_child - P_parent (N, 20, 2)
    rel_vecs = (kps[:, child_idx, :] - kps[:, parent_idx, :]) / box_scales.clamp(min=1e-6)
    bone_vecs_norm = F.normalize(rel_vecs, p=2, dim=2)
    
    pair_weights = torch.minimum(pose_scores[:, parent_idx], pose_scores[:, child_idx])
    pair_weights[pair_weights < 0.3] = 0.0    

    # loop        
    order = scores.argsort(0, descending=True)
    keep = torch.zeros(N, dtype=torch.int64, device=boxes.device)
    keep_idx = 0

    while order.numel() > 0:
        i = order[0]
        keep[keep_idx] = i
        keep_idx += 1
        if order.numel() == 1:
            break

        rest = order[1:]

        # iou
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)

        # only pose compare for iou > threshold
        candidate_mask = iou > iou_threshold
        if not candidate_mask.any():
            order = rest
            continue
            
        pose_reliable_mask = (valid_nums[i] >= 10) & (valid_nums[rest] >= 10)
        final_mask = torch.zeros_like(candidate_mask)

        active_pose_mask = candidate_mask & pose_reliable_mask
        if active_pose_mask.any():
            target_indices = rest[active_pose_mask]
            
            # point similarity (M, 20, 2) -> (M, 20)
            diff_rel = rel_vecs[i:i+1] - rel_vecs[target_indices] 
            point_norm = torch.norm(diff_rel, dim=2)
            point_dissim = 1.0 - torch.exp(-5.0 * (point_norm**2))

            # bone cosine similarity (1, 20, 2) * (M, 20, 2) -> (M, 20)
            cos_sim = torch.sum(bone_vecs_norm[i:i+1] * bone_vecs_norm[target_indices], dim=2)
            bone_dissim = 1.0 - (cos_sim + 1.0) / 2.0

            # weighted
            w_i = pair_weights[i:i+1] # (1, 20)
            w_target = pair_weights[target_indices] # (M, 20)
            combined_weights = torch.minimum(w_i, w_target).clamp(min=1e-6) # (M, 20)

            m_point_d = torch.sum(point_dissim * combined_weights, dim=1) / torch.sum(combined_weights, dim=1)
            m_bone_d = torch.sum(bone_dissim * combined_weights, dim=1) / torch.sum(combined_weights, dim=1)

            # Suppress ONLY if both point distances and bone angular differences are small
            suppress = (m_point_d < point_threshold) & (m_bone_d < bone_threshold)
            final_mask[active_pose_mask] = suppress
        
        fallback_mask = candidate_mask & (~pose_reliable_mask)
        if fallback_mask.any():
            final_mask[fallback_mask] = True

        order = rest[~final_mask]

    return keep[:keep_idx]