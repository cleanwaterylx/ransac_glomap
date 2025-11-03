import pycolmap
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

def invert_pose(q, t):
    """反转一个位姿：将 q, t 从 A->B 变成 B->A"""
    r_inv = R.from_quat(q).inv()
    t_inv = -r_inv.apply(t)
    return r_inv.as_quat(), t_inv

def compute_pose_error(image_pose_gt, image_pose):
    # R.from_quat(x ,y, z, w) 
    R_c2w_gt = R.from_quat(image_pose_gt[0]).as_matrix()
    t_c2w_gt = image_pose_gt[1]
    
    R_c2w = R.from_quat(image_pose[0]).as_matrix()
    t_c2w = image_pose[1]

    dt = np.linalg.norm(t_c2w_gt - t_c2w)
    cos = np.clip(((np.trace(R_c2w_gt @ R_c2w.T)) - 1) / 2, -1, 1)
    dR = np.rad2deg(np.abs(np.arccos(cos)))
    return dt, dR


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall

def compute_recall_at_threshold(errors, threshold):
    errors, recall = compute_recall(errors)
    idx = np.searchsorted(errors, threshold, side="right") - 1
    print(errors)
    if idx < 0:
        return 0.0
    return recall[idx] * 100


def compute_auc(errors, thresholds, min_error: Optional[float] = None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e)/t
        aucs.append(auc*100)
    return aucs


name = 'einstein_1'

# reconstruction = pycolmap.Reconstruction("cables_1/sparse/0")
reconstruction = pycolmap.Reconstruction(f"{name}/sparse_new_aligned")

imgs_glomap = {}
imgs_gt = {}
all_errors = {}
thresholds = [0.1, 0.2, 0.5]

for image_id, image in reconstruction.images.items():
    print("Image:", image.name)
    # print(invert_pose(image.cam_from_world().rotation.quat, image.cam_from_world().translation))
    imgs_glomap[image.name] = invert_pose(image.cam_from_world().rotation.quat, image.cam_from_world().translation) 

# 读取 img_pose_gt.txt
with open(f"{name}/img_pose_gt.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_name = parts[0]
        tx, ty, tz = map(float, parts[1:4])  # 只取前三个平移分量
        qx, qy, qz, qw = map(float, parts[4:8])  # 取后四个四元数分量
        imgs_gt[image_name] = (np.array([qx, qy, qz, qw]), np.array([tx, ty, tz]))

with open(f"{name}/rgb.txt", "r") as f:
    for line in f:
        time_str, img_name = line.split()
        img_name = img_name.replace("rgb/", "")  # 去掉前缀
        if img_name in imgs_glomap:
            error = compute_pose_error(imgs_gt[img_name], imgs_glomap[img_name])
            all_errors[img_name] = error
        else:
            error = (np.inf, 180)
            all_errors[img_name] = error

dt_errors = [dt for dt, _ in all_errors.values()]
dR_errors = [dR for _, dR in all_errors.values()]

print(f"{name}")

print("len(imgs_glomap):", len(imgs_glomap))
print("len(imgs_gt):", len(imgs_gt))

recall = compute_recall_at_threshold(dt_errors, 0.1)
print(" dt Recall@0.1:", recall)

auc = compute_auc(dt_errors, thresholds, min_error=0.001)
print(" dt AUC@0.1 0.2 0.5:", auc)
