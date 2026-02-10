import pycolmap
import numpy as np
import open3d as o3d
import colorsys

name = '/home/disk3_SSD/ylx/dataset_glg_sfm/cup/'
model_path = f"{name}/sparse_ransac2/0"
# model_path = '/home/disk3_SSD/ylx/dataset_dopp/arc_de_triomphe/sparse_doppelgangers_0.800/0'
reconstruction = pycolmap.Reconstruction(model_path)

print(f"Loaded {len(reconstruction.images)} images, "
      f"{len(reconstruction.cameras)} cameras, "
      f"{len(reconstruction.points3D)} points")

print(reconstruction.summary())


# points = []
# colors = []

# for p in reconstruction.points3D.values():
#     points.append(p.xyz)
#     colors.append(p.color / 255.0)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(points))
# pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# ply_path = f"{name}/sparse/0/points1.ply"
# pcd = o3d.io.read_point_cloud(ply_path)


def create_camera_geometry(T, scale=1, color=[0.8, 0, 0.8]):
    """创建一个小相机金字塔（或坐标系）用于可视化"""
    # 定义相机在自身坐标系下的 5 个点：光心 + 图像平面四角
    pts = np.array([
        [0, 0, 0],                    # 光心
        [-1,  0.75, 2],               # 左上
        [ 1,  0.75, 2],               # 右上
        [ 1, -0.75, 2],               # 右下
        [-1, -0.75, 2],               # 左下
    ]) * scale

    # 转换到世界坐标系
    pts_world = (T[:3, :3] @ pts.T + T[:3, 3:4]).T

    # 定义金字塔边线（光心到角点 + 矩形边框）
    lines = [
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def read_groups_from_file(file_path):
    clusters = []  # 用来存储所有 cluster
    with open(file_path, "r") as f:
        cluster = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Cluster"):
                # 如果已经有累积的 cluster，先存进去
                if cluster:
                    clusters.append(cluster)
                    cluster = []
            else:
                # 当前行是图片文件名列表
                cluster.extend(line.split())
        # 最后一组也要加进去
        if cluster:
            clusters.append(cluster)

    groups = []
    for c in clusters:
        # print(len(c), "Cluster:", c)
        groups.append(c)

    return groups


groups_1 = read_groups_from_file(f"{name}/image_clusters_louvain.txt")
# groups_2 = read_groups_from_file(f"{name}/image_clusters_pose.txt")
num_group1 = len(groups_1)
# num_group2 = len(groups_2)

def distinct_colors(n_groups):
    """生成 n_groups 个视觉上区别较大的 RGB 颜色"""
    colors = []
    for i in range(n_groups):
        hue = i / n_groups
        sat = 0.8
        val = 0.8 + 0.1 * ((-1) ** i)
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(rgb)
    return colors

group_colors_1 = distinct_colors(num_group1)
# group_colors_2 = distinct_colors(num_group2)

# 将图像名映射到组颜色
image_to_color_1 = {}
for color, group in zip(group_colors_1, groups_1):
    for img_name in group:
        image_to_color_1[img_name] = color

# image_to_color_2 = {}
# for color, group in zip(group_colors_2, groups_2):
#     for img_name in group:
#         image_to_color_2[img_name] = color



camera_meshes = []
for img in reconstruction.images.values():
    # PyCOLMAP: world_to_cam（R,t），我们要 world_from_cam
    name = img.name
    color_1 = image_to_color_1.get(name, [0.5, 0.5, 0.5])
    # color_2 = image_to_color_2.get(name, [0.5, 0.5, 0.5])
    # color_2 = [0, 0, 0]
    T_1 = np.eye(4)
    T_1[:3, :3] = img.cam_from_world().rotation.matrix().T
    T_1[:3, 3] = -img.cam_from_world().rotation.matrix().T @ img.cam_from_world().translation
    T_2 = np.eye(4)
    T_2[:3, :3] = img.cam_from_world().rotation.matrix().T
    T_2[:3, 3] = -img.cam_from_world().rotation.matrix().T @ img.cam_from_world().translation + np.array([5, 0, 0])  # 偏移一点，避免重叠
    camera_meshes.append(create_camera_geometry(T_1, color=color_1))
    # camera_meshes.append(create_camera_geometry(T_2, color=color_2))
    
o3d.visualization.draw_geometries([*camera_meshes])

# o3d.visualization.draw(
#     [pcd, *camera_meshes],
#     point_size=3,          # 缩小点
#     line_width=1,          # 调整相机线条
#     bg_color=(1,1,1,1),      # 背景黑色
#     show_skybox=False,
# )

