import pycolmap
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from typing import Optional

def invert_pose(q, t):
    """反转一个位姿：将 q, t 从 A->B 变成 B->A"""
    r_inv = R.from_quat(q).inv()
    t_inv = -r_inv.apply(t)
    return r_inv.as_quat(), t_inv

def compute_rotation_error(R_rotation_gt, R_rotation):
    # R.from_quat(x ,y, z, w) 
    # R_c2w_gt = R.from_quat(rel_rotation_gt).as_matrix()
    
    # R_c2w = R.from_quat(rel_rotation).as_matrix()

    cos = np.clip(((np.trace(R_rotation_gt @ R_rotation.T)) - 1) / 2, -1, 1)
    dR = np.rad2deg(np.abs(np.arccos(cos)))
    return dR

def compute_vec_error(t_gt, t):
    n = np.linalg.norm(t_gt) * np.linalg.norm(t)
    return np.rad2deg(np.arccos(np.clip(np.dot(t_gt, t) / n, -1.0, 1.0)))


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


name = 'living_room'
reconstruction_glomap = pycolmap.Reconstruction(f"/home/disk3_SSD/ylx/ETH3D-mvs-dslr_undistorted/{name}/images/sparse_ransac/0")
# print(reconstruction_glomap.summary())
# reconstruction_glomap = pycolmap.Reconstruction(f'/home/disk3_SSD/ylx/ETH3D-mvs-dslr/courtyard/dslr_calibration_undistorted')
reconstruction_gt = pycolmap.Reconstruction(f'/home/disk3_SSD/ylx/ETH3D-mvs-dslr_undistorted/{name}/dslr_calibration_undistorted')

imgs_glomap = {}
imgs_gt = {}
all_errors = {}

# 读取 img_glomap   pose c2w
for image_id, image in reconstruction_glomap.images.items():
    print("Image:", os.path.join('dslr_images_undistorted', image.name))
    print(invert_pose(image.cam_from_world().rotation.quat, image.cam_from_world().translation))
    # imgs_glomap[image.name] = invert_pose(image.cam_from_world().rotation.quat, image.cam_from_world().translation)
    imgs_glomap[os.path.join('dslr_images_undistorted', image.name)] = invert_pose(image.cam_from_world().rotation.quat, image.cam_from_world().translation) 


print('----------------')

# 读取 img_pose_gt
for image_id, image in reconstruction_gt.images.items():
    print("Image:", image.name)
    imgs_gt[image.name] = invert_pose(image.cam_from_world().rotation.quat, image.cam_from_world().translation)

print(len(imgs_glomap))
print(len(imgs_gt))


# 计算 relative rotation error
image_names = sorted(imgs_gt.keys())
for i in range(len(image_names)):
    for j in range(i + 1, len(image_names)):
        name_i = image_names[i]
        name_j = image_names[j]
        if name_i in imgs_glomap.keys() and name_j in imgs_glomap.keys():
            R_i_gt, t_i_gt = R.from_quat(imgs_gt[name_i][0]).as_matrix(), imgs_gt[name_i][1]
            R_j_gt, t_j_gt = R.from_quat(imgs_gt[name_j][0]).as_matrix(), imgs_gt[name_j][1]
            R_rel_gt = R_j_gt.T @ R_i_gt
            t_rel_gt =  R_j_gt.T @ (t_i_gt - t_j_gt)
            R_i_glomap, t_i_glomap = R.from_quat(imgs_glomap[name_i][0]).as_matrix(), imgs_glomap[name_i][1]
            R_j_glomap, t_j_glomap = R.from_quat(imgs_glomap[name_j][0]).as_matrix(), imgs_glomap[name_j][1]
            R_rel_glomap = R_j_glomap.T @ R_i_glomap
            t_rel_glomap = R_j_glomap.T @ (t_i_glomap - t_j_glomap)

            R_error = compute_rotation_error(R_rel_gt, R_rel_glomap)
            t_error = compute_vec_error(t_rel_gt, t_rel_glomap)
            all_errors[(name_i, name_j)] = max(R_error, t_error)
            # print(name_i, name_j)
            # print(R_error, t_error)
            # input()
        else:
            error = 180
            all_errors[(name_i, name_j)] = error

dR_errors = [dR for dR in all_errors.values()]


print("len(imgs_glomap):", len(imgs_glomap))
print("len(imgs_gt):", len(imgs_gt))
print(name)
# recall = compute_recall_at_threshold(dt_errors, 0.1)
# print(" dt Recall@0.1:", recall)

rotation_thresholds = [1, 3, 5]
auc = compute_auc(dR_errors, rotation_thresholds, min_error=0.001)
print(" dt AUC@1, 3, 5:", auc)
