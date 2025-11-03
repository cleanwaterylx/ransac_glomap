import pycolmap
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def invert_pose(q, t):
    """反转一个位姿：将 q, t 从 A->B 变成 B->A"""
    r_inv = R.from_quat(q).inv()
    t_inv = -r_inv.apply(t)
    return r_inv.as_quat(), t_inv

name = 'camera_shake_2'

imgs_glomap = {}
imgs_gt = {}
reconstruction = pycolmap.Reconstruction(f"{name}/sparse_aligned")

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


def pose_to_transform(q, t):
    """把四元数+平移 转成 4x4 变换矩阵"""
    rot = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T

def create_camera_frame(q, t, size=0.2, color=[1, 0, 0]):
    """创建一个相机坐标系（用坐标系网格表示）"""
    T = pose_to_transform(q, t)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    frame.paint_uniform_color(color)  # 坐标轴仍然保留 RGB，但整体 mesh 有 tint
    return frame

vis_objs = []

# 可视化 GLOMAP 位姿 (红色)
for name, (q, t) in imgs_glomap.items():
    vis_objs.append(create_camera_frame(q, t, size=0.02, color=[1, 0, 0]))

# 可视化 GT 位姿 (绿色)
for name, (q, t) in imgs_gt.items():
    vis_objs.append(create_camera_frame(q, t, size=0.02, color=[0, 1, 0]))

# 打开窗口
o3d.visualization.draw_geometries(vis_objs)

