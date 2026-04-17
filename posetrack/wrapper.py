import numpy as np


class DetsWrapper:
    def __init__(self, xywh, conf, cls, xyxy):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls
        self.xyxy = xyxy

class PosesWrapper:
    def __init__(self, xy, conf):
        self.xy = xy
        self.conf = conf

def run_pose_tracker_wrapper(tracker, img_cv2, bboxes_xyxy, scores, clses, kps_xy, kps_conf):
    """
    Inputs:
        tracker: PoseTracker
        img_cv2: image
        bboxes_xyxy: list or ndarray, shape (N, 4)
        scores: list or ndarray, shape (N,)
        clses: list or ndarray, shape (N,) -> 0 left, 1 right
        kps_xy: list or ndarray, shape (N, 21, 2)
        kps_conf: list or ndarray, shape (N, 21)
        
    Return:
        parsed_tracks: list of dict of each track.
    """
    N = len(bboxes_xyxy)

    if N > 0:
        bboxes_xyxy = np.array(bboxes_xyxy, dtype=np.float32)
        w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
        h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
        cx = bboxes_xyxy[:, 0] + w / 2.0
        cy = bboxes_xyxy[:, 1] + h / 2.0
        bboxes_xywh = np.stack([cx, cy, w, h], axis=1)
        
        dets = DetsWrapper(
            xywh=bboxes_xywh, 
            conf=np.array(scores, dtype=np.float32), 
            cls=np.array(clses, dtype=np.float32), 
            xyxy=bboxes_xyxy
        )
        poses = PosesWrapper(
            xy=np.array(kps_xy, dtype=np.float32), 
            conf=np.array(kps_conf, dtype=np.float32)
        )
    else:
        dets = DetsWrapper(np.empty((0,4)), np.empty((0,)), np.empty((0,)), np.empty((0,4)))
        poses = PosesWrapper(np.empty((0,21,2)), np.empty((0,21)))


    tracked_results = tracker.update(dets, poses, img_cv2, feats=None)
    
    parsed_tracks = []
    for res in tracked_results:
        # Tracker result: [x1, y1, x2, y2, track_id, score, cls, idx, kps(42), kps_score(21)]
        x1, y1, x2, y2 = res[0:4]
        track_id = int(res[4])
        cls = int(res[6]) 
        kps_42 = res[8:50] 
        
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        
        parsed_tracks.append({
            'track_id': track_id,
            'handedness': cls,
            'bbox_cxcywh': np.array([cx, cy, w, h]),
            'kps_flattened': np.array(kps_42)
        })
        
    return parsed_tracks