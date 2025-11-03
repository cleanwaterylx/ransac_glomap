from pdb import run
import re
from collections import defaultdict, deque
from turtle import pen
from typing import final
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import time
import logging
import subprocess
import shutil
import pycolmap
import os
import copy
import cv2
import sys
from plyfile import PlyData, PlyElement
import struct


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                if error > 2.0:
                    continue
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1
    xyzs = np.delete(xyzs, np.arange(count,num_points),axis=0)
    rgbs = np.delete(rgbs, np.arange(count,num_points),axis=0)
    errors = np.delete(errors, np.arange(count,num_points),axis=0)
    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        count = 0
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            if error > 2.0 or track_length < 3:
                continue
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1
    xyzs = np.delete(xyzs, np.arange(count,num_points),axis=0)
    rgbs = np.delete(rgbs, np.arange(count,num_points),axis=0)
    errors = np.delete(errors, np.arange(count,num_points),axis=0)
    return xyzs, rgbs, errors

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

rng = random.Random(10)
logging.basicConfig(level=logging.DEBUG)

imgs = {}
imgs_features = {} #  dic{image_name : features}
imgs_matches = {} # 
groups_imgs = []  # c2w pose
edge_weights = {}
img_name2id = {}
delete_edges = []
# K = np.array([
#             [3409, 0, 3114],
#             [0, 3409, 2066],
#             [0, 0, 1]
#         ], dtype=np.float32)

def run_command(command):
    try:
        # 使用 subprocess 捕获命令行输出
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        # process = subprocess.Popen(command, stdout=subprocess.STDOUT, stderr=subprocess.PIPE, shell=True, text=True, encoding='utf-8')

        # 实时获取输出并处理
        for line in iter(process.stderr.readline, ''):
            print(line, end='')  # 可以在这里发射信号将信息返回给调用者
            yield line.strip()   # 使用生成器来逐行返回输出内容
        
        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        yield f"Command failed with code {e.returncode}"

def colmap_run_feature_extractor(colmap_db_path, image_path, camera_model='PINHOLE'):
    print("running feature extracting")

    feat_extracton_cmd = 'colmap feature_extractor' + '\
            --database_path ' + colmap_db_path + '\
            --image_path ' + image_path + ' \
            --ImageReader.camera_model PINHOLE  \
            --ImageReader.single_camera 1 \
            --SiftExtraction.use_gpu 1 '

    print(feat_extracton_cmd)
    
    for line in run_command(feat_extracton_cmd):
        print(line)

    print("feature extracting done")

def colmap_run_feature_matcher(colmap_db_path):
    print("running feature matching")
    
    feat_matching_cmd = 'colmap exhaustive_matcher'+ '\
            --database_path ' + colmap_db_path 
    
    print(feat_matching_cmd)

    for line in run_command(feat_matching_cmd):
        print(line)
    
    print("feature matching done")

def glomap_run_mapper(colmap_db_path, image_path, output_path):
    print("running glomap mapper")

    glomap_mapper_cmd = 'glomap mapper ' + '\
            --database_path ' + colmap_db_path + '\
            --image_path '  + image_path + '\
            --output_path '  + output_path + '\
            --output_format txt'
    # os.mkdir(output_path)
    # glomap_mapper_cmd = 'colmap mapper ' + '\
    #         --database_path ' + colmap_db_path + '\
    #         --image_path '  + image_path + '\
    #         --output_path '  + output_path 
    
    for line in run_command(glomap_mapper_cmd):
        print(line)

    print("glomap mapper done")

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"运行时间：{self.interval:.6f} 秒")
        
def parse_image_map(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            img_id, image_name = line.split(' ')
            img_name2id.update({image_name.strip(): int(img_id)})
            imgs.update({image_name.strip() : {
                "id" : int(img_id),
                "pose" : (np.array([0, 0, 0, 1]), np.zeros(3))
            }})
            

# 读取 view_graph.txt 并建立图
def parse_view_graph(file_path):
    graph = defaultdict(dict)
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            img, rest = line.split(":", 1)
            img = re.match(r'([^\[\]]+)', img.strip()).group(1)
            matches = re.findall(r'(\S+)\(([^)]+)\)', rest)
            for neighbor, pose in matches:
                quat_str, trans_str = pose.split(',')
                quat_vals = list(map(float, re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', quat_str)))
                trans_vals = list(map(float, trans_str.strip().split()))
                if len(quat_vals) == 4 and len(trans_vals) == 3:
                    q = np.array(quat_vals)  # (x, y, z, w)
                    t = np.array(trans_vals)  # (tx, ty, tz)
                    graph[img][neighbor] = (q, t)
    return graph

def get_index(img_name):
    # 提取数字部分，例如 '00001.png' -> 1
    return int(img_name.replace('.png', ''))

def set_edge_weight(graph):
    edge_weights = {}

    for img, neighbors in graph.items():
        idx = get_index(img)
        for neighbor in neighbors:
            neighbor_idx = get_index(neighbor)
            # 设置权重为 0，如果是相邻图像（i 和 i+1）
            if abs(neighbor_idx - idx) == 1:
                weight = 0
            else:
                weight = 1
            # 注意：加上双向边
            edge_weights[(img, neighbor)] = weight
            edge_weights[(neighbor, img)] = weight

    return edge_weights

   
def quat_mul(q1, q2):
    """四元数乘法"""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

def invert_pose(q, t):
    """反转一个位姿：将 q, t 从 A->B 变成 B->A"""
    r_inv = R.from_quat(q).inv()
    t_inv = -r_inv.apply(t)
    return r_inv.as_quat(), t_inv

def compose_pose(q1, t1, q2, t2):
    """
        组合两个位姿,得到的2下 w2c

        P2*P1
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_new = r2 * r1
    t_new = r2.apply(t1) + t2
    return r_new.as_quat(), t_new

def pose_to_matrix(q, t):
    """
    将四元数和平移向量转换为 4x4 齐次变换矩阵。
    
    参数:
        q: 四元数，形如 [x, y, z, w]
        t: 平移向量，形如 [tx, ty, tz]
    
    返回:
        4x4 齐次变换矩阵 numpy.ndarray
    """
    P = np.eye(4)
    R_mat = R.from_quat(q).as_matrix()  # 3x3 旋转矩阵
    P[:3, :3] = R_mat
    P[:3, 3] = t
    return P

def pose_to_rotvec_t(P):
    R_mat = P[:3, :3]
    t = P[:3, 3]
    r = R.from_matrix(R_mat).as_rotvec()
    return r, t

def pose_to_q_t(P):
    R_mat = P[:3, :3]
    t = P[:3, 3]
    q = R.from_matrix(R_mat).as_quat()
    return q, t

def rotation_error_axis_angle(q1, q2):
    """两个四元数之间的旋转误差（轴角）"""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_diff = r1.inv() * r2
    return r_diff.as_rotvec()


def apply_R_to_t(R, t):
    t = np.asarray(t).reshape(3, 1)
    t_new = R @ t
    return t_new

def matrix_to_Rt(matrix):
    R_mat = matrix[:3, :3]
    t = matrix[:3, 3]
    return R_mat, t

def matrix_to_qt(matrix):
    R_mat = matrix[:3, :3]
    q = R.as_quat()
    t = matrix[:3, 3]
    return q, t

def solve_pnp(graph, edges, group1, group2, i, j):
    group1_name = f'group_{i}'
    rec_path1 = f'{name}/groups/{group1_name}/sparse/0'
    group2_name = f'group_{j}'
    rec_path2 = f'{name}/groups/{group2_name}/sparse/0'
    if not os.path.exists(rec_path1) or not os.path.exists(rec_path2):
        print(f"Reconstruction path does not exist: {rec_path1} or {rec_path2}")
        return {}, edges
    
    reconstruction1 = pycolmap.Reconstruction(rec_path1)
    reconstruction2 = pycolmap.Reconstruction(rec_path2)

    
    success_edges = {}
    fail_edges = []
    
    for img1, img2 in edges:
        # print(img1, img2)
        # input()
        # print(np.linalg.inv(pose_to_matrix(graph[img1][img2][0], graph[img1][img2][1])))
        img_matches = imgs_matches[(img1, img2)]
        feature_idx1 = list(img_matches.keys())
        feature_idx2 = list(img_matches.values())
        # print(len(feature_idx1), len(feature_idx2))
        # print(feature_idx1)
        # print(feature_idx2)

        img1_points2D = get_points2D_by_feature_idx(img1, feature_idx1)
        img2_points2D = get_points2D_by_feature_idx(img2, feature_idx2)    # 2d points

        points3D, points2D = get_2d3d_correspondences_from_3dpoint(reconstruction1, img1)
        if len(points3D) == 0:
            fail_edges.append((img1, img2))
            continue

        points2D_3D_idx, points2D_match_idx = get_points2D_3D_idx_by_points2D_match(img1_points2D, points2D)
        # print(points2D_3D_idx)
        # print(len(img1_points2D), len(points2D))
        pt3d = points3D[points2D_3D_idx]    # 3d points
        pt2d = img2_points2D[points2D_match_idx]
        # print(points3D)
        # print(points3D.shape)

        # 从外部输入 K
        K = get_K(reconstruction2, img2)
        # print(pt3d.shape, pt2d.shape)
        # cv2 return w2c
        if len(pt3d) >= 4 and len(pt2d) >= 4 and len(pt3d) == len(pt2d):
            success, rvec, tvec, inliers = cv2.solvePnPRansac(pt3d, pt2d, K, None, iterationsCount=100, reprojectionError=8.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
            print('rvec tvec success', rvec.T, tvec.T, success)
            R_mat, _ = cv2.Rodrigues(rvec)
            tvec = -R_mat.T @ tvec
            R_mat = R_mat.T
            # print(success)
            # print(R, tvec.reshape((1, 3)), len(inliers))
            if success and not np.any(np.isnan(R_mat)) and not np.any(np.isnan(tvec)) and len(inliers) >= 4:
                success_edges[(img1, img2)] = (R.from_matrix(R_mat).as_quat(), tvec.reshape(-1))
                # print(group1['imgs'][img1]['pose'])
                # print(success_edges[(img1, img2)])
                # input()
            else:
                fail_edges.append((img1, img2))
        else:
            fail_edges.append((img1, img2))

    return success_edges, fail_edges


def solve_scales_new(graph, sample_edges, group1, group2, success_edges):

    # print('success_edges', success_edges)
    # print('fail_edges', fail_edges)
    
    # for a, b in edges:
    #     pose_group1_mean_to_b = np.linalg.inv(pose_to_matrix(group1['imgs'][a]['pose'][0], group1['imgs'][a]['pose'][1])) @ 

    n = len(sample_edges)
    T1 = np.zeros((3, n))
    T2 = np.zeros((3, n))
    A = np.zeros((3 * n, 1))
    B = np.zeros((3 * n, 1))


    for idx, (img1, img2) in enumerate(sample_edges):
        pose_group1_mean_to_b = np.linalg.inv(pose_to_matrix(success_edges[(img1, img2)][0], success_edges[(img1, img2)][1])) @ pose_to_matrix(group1['q_mean'], group1['t_mean'])
        pose_b_to_group2_mean = np.linalg.inv(pose_to_matrix(group2['q_mean'], group2['t_mean'])) @ pose_to_matrix(group2['imgs'][img2]['pose'][0], group2['imgs'][img2]['pose'][1])
        R_1b, t_1b = matrix_to_Rt(pose_group1_mean_to_b)
        R_b2, t_b2 = matrix_to_Rt(pose_b_to_group2_mean)

        T1[:, idx : idx + 1] = apply_R_to_t(R_b2, t_1b)
        T2[:, idx : idx + 1] = t_b2.reshape(3, 1)

    for idx, (img1, img2) in enumerate(sample_edges):
        A[idx * 3 : (idx + 1) * 3, 0] = n * T2[:, idx] - T2.sum(axis=1)
        B[idx * 3 : (idx + 1) * 3, 0] = n * T1[:, idx] - T1.sum(axis=1)

    # print("T2", T2)
    # print("A", A)
    # print("B", B)

    # print('A.T @ B', A.T @ B)
    # print('A.T @ A', A.T @ A)
    beta = -np.linalg.lstsq(A.T @ A, A.T @ B, rcond=None)[0]
    # print((A.T @ B).shape, (A.T @ A).shape)
    # beta = -A.T @ B / A.T @ A
    # print('beta = ', beta)
    # beta = np.array([[65]])
    residual = (A @ beta).reshape(-1, 1) + B
    # print(residual)
    # print("residual norm = ", np.linalg.norm(residual))
    # input()

    return beta.reshape(-1)
    

def get_K(reconstruction, image_name):
    image = None
    for img in reconstruction.images.values():
        if img.name == image_name:
            image = img
            break

    if image is None:
        raise ValueError(f"Image {image_name} not found in reconstruction.")
    
    camera = image.camera
    pramas = camera.params
    focal_length_idxs = camera.focal_length_idxs()
    principal_point_idxs = camera.principal_point_idxs()
    fx = pramas[focal_length_idxs[0]]  
    fy = pramas[focal_length_idxs[1]]
    cx = pramas[principal_point_idxs[0]]
    cy = pramas[principal_point_idxs[1]]
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ])

    return K


def scale_pose(graph, group, edges, beta):
    tmp_group = copy.deepcopy(group) 
    for node in group['nodes']:
        q, t = group['imgs'][node]['pose']
        tmp_group['imgs'][node]['pose'] = (q, beta * t)  # 缩放group
    tmp_group['t_mean'] = beta * tmp_group['t_mean']

    # print(tmp_group)

    return tmp_group

def ransac_edges(graph, edges, group1, group2, i, j, max_iter=100):
    
    # a, b = edge
    # q_gt, t_gt = graph[a][b]

    best_inliers = []
    best_avg_q = None
    best_avg_t = None

    success_edges, fail_edges = solve_pnp(graph, edges, group1, group2, i, j)
    # delete fail_edges
    delete_edges.extend(fail_edges)
    print('success_edges', success_edges)
    if len(success_edges) < 5:
        for edge in success_edges.keys():
            delete_edges.append(edge)
        return

    for iter in range(max_iter):
        num_samples = int(len(success_edges)*0.4)
        sample_edges = rng.sample(list(success_edges), num_samples)
        # samples = [('00006.png', '00008.png'), ('00006.png', '00009.png'), ('00007.png', '00008.png'), ('00007.png', '00009.png'), ('00006.png', '00010.png'), ('00007.png', '00011.png')]
        # samples = [('00004.png', '00008.png'), ('00007.png', '00010.png'), ('00005.png', '00009.png')]
        # samples = [('00004.png', '00008.png'), ('00006.png', '00008.png'), ('00007.png', '00009.png')]
        # print(samples)

        print('samples: ', sample_edges)
        beta = solve_scales_new(graph, sample_edges, group1, group2, success_edges)
        print(f'try {iter + 1} times', beta.shape, beta)
        if np.isnan(beta[0]) or beta[0] < 1e-6:
            continue
        
        # input()
        # for i, edge in enumerate(samples):
        #     print(edge, x[i])
        # input()

        # print(group2)
        tmp_group2 = scale_pose(graph, group2, sample_edges, beta[0])
        # print('tmp_group2', tmp_group2)
        # input()

        # sample_edges -> avg_q, avg_t
        rel_pose = []
        for (img1, img2) in sample_edges:
            # all pose c2w
            # P_group1mean_2_b  = P_a*P_mean-1
            pose_group1_mean_to_b = np.linalg.inv(pose_to_matrix(success_edges[(img1, img2)][0], success_edges[(img1, img2)][1])) @ pose_to_matrix(group1['q_mean'], group1['t_mean'])

            # P_b_2_group2mean
            pose_b_to_group2_mean = np.linalg.inv(pose_to_matrix(tmp_group2['q_mean'], tmp_group2['t_mean'])) @ pose_to_matrix(tmp_group2['imgs'][img2]['pose'][0], tmp_group2['imgs'][img2]['pose'][1])

            rel_pose.append((pose_b_to_group2_mean @ pose_group1_mean_to_b, (img1, img2)))

        # for pose in rel_pose:
        #     print('rel_pose', pose) 
        # input()

        rotvecs = [pose_to_rotvec_t(pose[0])[0] for pose in rel_pose]
        translations = [pose_to_rotvec_t(pose[0])[1] for pose in rel_pose]
        rotvec_mean = np.mean(rotvecs, axis=0)
        # print('translations', translations)
        # input()
        t_mean = np.mean(translations, axis=0)
        r_mean = R.from_rotvec(rotvec_mean)
        q_mean = r_mean.as_quat()

        inliers = []
        # rel_pose_qt = [(pose_to_q_t(pose[0]), pose[1]) for pose in rel_pose]   # (q,t,(a,b))
        for img1, img2 in success_edges.keys():
            pose_group1_mean_to_b = np.linalg.inv(pose_to_matrix(success_edges[(img1, img2)][0], success_edges[(img1, img2)][1])) @ pose_to_matrix(group1['q_mean'], group1['t_mean'])

            # P_b_2_group2mean
            pose_b_to_group2_mean = np.linalg.inv(pose_to_matrix(tmp_group2['q_mean'], tmp_group2['t_mean'])) @ pose_to_matrix(tmp_group2['imgs'][img2]['pose'][0], tmp_group2['imgs'][img2]['pose'][1])

            q, t = pose_to_q_t(pose_b_to_group2_mean @ pose_group1_mean_to_b)

            # print(img1, img2, pose_to_matrix(success_edges[(img1, img2)][0], success_edges[(img1, img2)][1]))
            rotation_error = np.rad2deg(np.linalg.norm(rotation_error_axis_angle(q, q_mean)))    # degrees
            trans_error = np.linalg.norm(t - t_mean)
            print(q, q_mean)
            # print(img1, img2)
            print(f"角度误差: {rotation_error:.3f}, 平移误差: {trans_error:.3f}")
            # input()
            if rotation_error < 5 and trans_error < 0.05 * np.linalg.norm(t_mean):  # 阈值可调  0.05 0.5
                inliers.append((img1, img2))
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_avg_q = q_mean
            best_avg_t = t_mean
    
    # print(best_inliers)
    # print(len(best_inliers), "个内点")
    # input()

    if not best_inliers:
        print("没有找到足够的内点")
        # input()
        for edge in edges:
            delete_edges.append(edge)
        return

    best_inliers_set = set(best_inliers)
    outliers = set(success_edges) - best_inliers_set
    for outlier in outliers:
        a, b = outlier
        delete_edges.append((a, b))
    print('len(edges) = ', len(edges), 'len(success_edges)', len(success_edges), 'len(best_inliers)', len(best_inliers))
    print('outliers ', outliers)
    # input()
    
def run_glomap_for_groups(groups):
    for idx, group in enumerate(groups):
        if len(group['nodes']) == 1:
            print(f'Skip group_{idx} with only one image')
            continue
        group_name = f'group_{idx}'
        rec_path = f'{name}/groups/{group_name}'
        os.makedirs(rec_path, exist_ok=True)
        
        db_name = f'{group_name}.db'
        db_path = os.path.join(rec_path, db_name)
        image_path = os.path.join(rec_path, 'input')
        output_path = os.path.join(rec_path, 'sparse')
        
        print(f"Processing {group_name}")
        print(f"Database path: {db_path}")
        
        colmap_run_feature_extractor(db_path, image_path)
        colmap_run_feature_matcher(db_path)
        glomap_run_mapper(db_path, image_path, output_path)
    # idx = 7
    # group_name = f'group_{idx}'
    # rec_path = f'groups/{group_name}'
    # os.makedirs(rec_path, exist_ok=True)
    
    # db_name = f'{group_name}.db'
    # db_path = os.path.join(rec_path, db_name)
    # image_path = os.path.join(rec_path, 'input')
    # output_path = os.path.join(rec_path, 'sparse')
    
    # print(f"Processing {group_name}")
    # print(f"Database path: {db_path}")
    
    # colmap_run_feature_extractor(db_path, image_path)
    # colmap_run_feature_matcher(db_path)
    # glomap_run_mapper(db_path, image_path, output_path)

def has_files(folder_path: str) -> bool:
    """
    判断指定文件夹下是否存在文件（忽略子目录）

    参数:
        folder_path (str): 文件夹路径

    返回:
        bool: 如果有文件返回 True，否则 False
    """
    if not os.path.isdir(folder_path):
        return False
    
    return any(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path))

def read_abs_pose_from_glomap(group_name, group):
    rec_path = f'{name}/groups/{group_name}'
    output_path = os.path.join(rec_path, 'sparse/0')
    group_imgs = {}

    if len(group['nodes']) == 1:
        group_imgs.update({group['nodes'][0] : {'pose' : (np.array([0, 0, 0, 1]), np.zeros(3))}})
        imgs[group['nodes'][0]]['pose'] = (np.array([0, 0, 0, 1]), np.zeros(3))
        group['imgs'] = group_imgs
        return group_imgs
    
    if has_files(output_path):
        reconstruction = pycolmap.Reconstruction(output_path)
        print(reconstruction.summary())
    else:
        print(f"Skipping {group_name} as no reconstruction files found in {output_path}")
        return {}

    ply_path = os.path.join(output_path, "points3D.ply")
    bin_path = os.path.join(output_path, "points3D.bin")
    txt_path = os.path.join(output_path, "points3D.txt")

    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    for img_id, img in reconstruction.images.items():
        # print(img_id, img.name, img.cam_from_world.rotation.quat, img.cam_from_world.translation)
        
        # convert to c2w
        imgs[img.name]['pose'] = invert_pose(img.cam_from_world().rotation.quat, img.cam_from_world().translation)
        group_imgs.update({img.name : {'pose' : invert_pose(img.cam_from_world().rotation.quat, img.cam_from_world().translation)}})
        print(invert_pose(img.cam_from_world().rotation.quat, img.cam_from_world().translation))

    group['imgs'] = group_imgs

    return group_imgs

# image_a的2d-3d对应点
def get_2d3d_correspondences_from_3dpoint(reconstruction, image_name):
    image = None
    for img in reconstruction.images.values():
        if img.name == image_name:
            image = img
            break

    if image is None:
        raise ValueError(f"Image {image_name} not found in reconstruction.")
    
    # 使用字典去重
    point2d3d_dict = {}

    for point2D in image.points2D:
        if point2D.has_point3D():
            key = tuple(np.round(point2D.xy, decimals=10))  # 将二维点坐标作为key（注意精度）
            if key not in point2d3d_dict:
                pt3D = reconstruction.points3D[point2D.point3D_id]
                point2d3d_dict[key] = pt3D.xyz

    # 拆分为列表
    points2D = np.array(list(point2d3d_dict.keys()), dtype=np.float64)
    points3D = np.array(list(point2d3d_dict.values()), dtype=np.float64)

    return points3D, points2D

def parse_image_features(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue  # 跳过注释和空行
            
            parts = line.split()
            image_id_colon = parts[0]  # 例如 "1:"
            image_name = parts[1]      # 例如 "image001.jpg"
            feature_values = parts[2:] # 一连串 x1 y1 x2 y2 ...

            if len(feature_values) % 2 != 0:
                raise ValueError(f"Feature values count not even in line: {line}")

            coords = np.array([float(v) for v in feature_values], dtype=np.float64)
            features = coords.reshape(-1, 2)  # 每两个数字一组变成 N×2 的数组

            imgs_features[image_name] = features

# todo tol=1e-2 在模型较大较小时需要调整
def get_points2D_3D_idx_by_points2D_match(points2D_match, points2D_3D, tol=1e-2):
    point2D_3D_indices = []
    points2D_match_indices = []
    for idx, pt in enumerate(points2D_match):
        dists = np.linalg.norm(points2D_3D - pt, axis=1)  # 与所有特征的距离
        min_dist_idx = np.argmin(dists)
        # print(dists[min_dist_idx])
        if dists[min_dist_idx] < tol:
            point2D_3D_indices.append(min_dist_idx)
            points2D_match_indices.append(idx)

    return point2D_3D_indices, points2D_match_indices

def parse_image_matches(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or line == "":
            i += 1
            continue

        # 判断是否是新的一对图像匹配开始行（带冒号）
        if ":" in line:
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError(f"Unexpected match header format in line: {line}")

            image_name1 = tokens[2]
            image_name2 = tokens[3]
            i += 1

            matches = []
            while i < len(lines):
                match_line = lines[i].strip()

                if match_line == "" or ":" in match_line:
                    break  # 下一组开始了

                pair = match_line.split()
                if len(pair) != 2:
                    raise ValueError(f"Invalid match line: {match_line}")
                matches.append([int(pair[0]), int(pair[1])])
                i += 1

            if matches:
                match_dict_12 = {int(m[0]): int(m[1]) for m in matches}
                match_dict_21 = {int(m[1]): int(m[0]) for m in matches}

                imgs_matches[(image_name1, image_name2)] = match_dict_12
                imgs_matches[(image_name2, image_name1)] = match_dict_21
        else:
            i += 1

def get_points2D_by_feature_idx(image_name, feature_idx):
    features = imgs_features[image_name]

    return features[feature_idx]

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
        print(len(c), "Cluster:", c)
        group = {'nodes': c, 'q_mean': np.array([0, 0, 0, 1]), 't_mean': np.zeros(3), 'imgs': {}}
        groups.append(group)

    return groups

def filter_sequential_edges(edges):
    """
    输入: edges = [(file1, file2), ...]
    输出: 过滤掉连续编号的边
    """
    def get_num(name):
        # 提取数字部分
        return int(name.split('.')[0])

    kept_edges = []
    for u, v in edges:
        num_u, num_v = get_num(u), get_num(v)
        if abs(num_u - num_v) != 1:  # 连续编号 -> 跳过
            kept_edges.append((u, v))
    return kept_edges

def filter_sequential_edges_1(edges, mapping):
    """
    输入: edges = [(file1, file2), ...]
    输出: 过滤掉连续编号的边
    """
    def get_num(name):
        # 提取数字部分
        return int(name.split('.')[0])

    kept_edges = []
    for u, v in edges:
        num_u, num_v = get_num(mapping[u]), get_num(mapping[v])
        if abs(num_u - num_v) != 1:  # 连续编号 -> 跳过
            kept_edges.append((u, v))
        else:
            print('remove sequential edge', u, v)
    return kept_edges

def parse_rgb_file(rgb_txt_path):
    mapping = {}
    with open(rgb_txt_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 2:
            continue  # 跳过不符合格式的行
        _, filepath = parts
        filename = filepath.split("/")[-1]  # 去掉路径，只保留文件名
        new_name = f"{idx:05d}.png"
        mapping[filename] = new_name
    return mapping


name = '/home/disk3_SSD/ylx/data/28'

if __name__ == "__main__":

    view_graph_file_path = f'{name}/sparse/view_graph_after_relpose.txt'
    image_map_file_path = f'{name}/sparse/image_map.txt'
    image_matches_file_path = f'{name}/sparse/image_matches.txt'
    image_feature_file_path = f'{name}/sparse/image_feature.txt'
    # start_img = '00067.png'
    # end_img = '00089.png'
    start_img = '00053.png'
    end_img = '00055.png'

    parse_image_map(image_map_file_path)
    graph = parse_view_graph(view_graph_file_path)
    # edge_weights = set_edge_weight(graph)

    # read image_feature.txt
    parse_image_features(image_feature_file_path)
    # print(imgs_features)
    # input()

    # read image_matches.txt
    parse_image_matches(image_matches_file_path)
    # print(imgs_matches[('00000.png', '00001.png')])
    # input()
    

    num_edges = sum(len(edges) for edges in graph.values())
    print(f"初始图中共有 {len(graph)} 个节点，{num_edges} 条边")

    # read groups from file
    groups = read_groups_from_file(f"{name}/image_clusters.txt")

    # groups = []
    # groups.append({'nodes':['00000.png', '00001.png', '00002.png', '00003.png', '00004.png', '00005.png', '00006.png', '00007.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # # groups.append({'nodes':['00008.png', '00009.png', '00010.png', '00011.png', '00012.png', '00013.png', '00014.png', '00015.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00006.png', '00007.png', '00008.png', '00009.png', '00010.png', '00011.png', '00012.png', '00013.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00016.png', '00017.png', '00018.png', '00019.png', '00020.png', '00021.png', '00022.png', '00023.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00024.png', '00025.png', '00026.png', '00027.png', '00028.png', '00029.png', '00030.png', '00031.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00032.png', '00033.png', '00034.png', '00035.png', '00036.png', '00037.png', '00038.png', '00039.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00039.png', '00040.png', '00041.png', '00042.png', '00043.png', '00044.png', '00045.png', '00046.png', '00047.png', '00048.png', '00049.png', '00050.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00050.png', '00051.png', '00052.png', '00053.png', '00054.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00061.png', '00062.png', '00063.png', '00064.png', '00065.png', '00066.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00054.png', '00055.png', '00056.png', '00057.png', '00058.png', '00059.png', '00060.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00067.png', '00068.png', '00069.png', '00070.png', '00071.png', '00072.png', '00073.png', '00074.png', '00075.png', '00076.png', '00077.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00078.png', '00079.png', '00080.png', '00081.png', '00082.png', '00083.png', '00084.png', '00085.png', '00086.png', '00087.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00088.png', '00089.png', '00090.png', '00091.png', '00092.png', '00093.png', '00094.png', '00095.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00096.png', '00097.png', '00098.png', '00099.png', '00100.png', '00101.png', '00102.png', '00103.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00104.png', '00105.png', '00106.png', '00107.png', '00108.png', '00109.png', '00110.png', '00111.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00112.png', '00113.png', '00114.png', '00115.png', '00116.png', '00117.png', '00118.png', '00119.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00120.png', '00121.png', '00122.png', '00123.png', '00124.png', '00125.png', '00126.png', '00127.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00128.png', '00129.png', '00130.png', '00131.png', '00132.png', '00133.png', '00134.png', '00135.png'], 'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00136.png', '00137.png', '00138.png', '00139.png', '00140.png', '00141.png'],                           'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    # groups.append({'nodes':['00142.png', '00143.png', '00144.png', '00145.png'],                                                     'q_mean':np.array([0,0,0,1]), 't_mean':np.zeros(3), 'imgs' : {}})
    
    # copy images to group folders
    shutil.rmtree(f'{name}/groups', ignore_errors=True)  # 删除已存在的 groups 文件夹
    for idx, group in enumerate(groups):
        target_input_dir = os.path.join(f'{name}/groups/group_{idx}', 'input')
        os.makedirs(target_input_dir, exist_ok=True)  # 创建目标文件夹

        for filename in group['nodes']:
            src_path = os.path.join(f'{name}/input', filename)
            dst_path = os.path.join(target_input_dir, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f'Copied {src_path} -> {dst_path}')
            else:
                print(f'Warning: {src_path} does not exist.')
    input()

    # # skip the glomap for groups if you have already run it
    run_glomap_for_groups(groups)

    print('glomap for groups done!')
    input()

    new_groups = []

    for idx, group in enumerate(groups):
        # if idx == 14 or idx == 15: continue
        group_name = f'group_{idx}'
        nodes = group['nodes']
        print(group_name)
        group_imgs = read_abs_pose_from_glomap(group_name, group)

        if not group_imgs:
            print(f"Skipping group_{idx} as no poses were read.")
            continue

        new_nodes = []
        for im in nodes:
            if im in group_imgs:
                new_nodes.append(im)
            else:
                print(f"Warning: Image {im} in group_{idx} has no pose. Skipping this group.")
        group['nodes'] = new_nodes
                
        rotvecs = [R.from_quat(group_imgs[im]["pose"][0]).as_rotvec() for im in group['nodes']]
        translations = [group_imgs[im]["pose"][1] for im in group['nodes']]
        print(len(rotvecs))
        # input()
        rotvec_mean = np.mean(rotvecs, axis=0)
        r_mean = R.from_rotvec(rotvec_mean)
        q_mean = r_mean.as_quat()   # c2w
        t_mean = np.mean(translations, axis=0)   # c2w
        group['q_mean'] = q_mean
        group['t_mean'] = t_mean
        new_groups.append(group)
        print('----------------------------------------------------------------------------------')
        # print(t_mean)
        # print(translations)
        # input()
        
    groups = new_groups

    for idx, group in enumerate(groups):
        print('group name', f'group_{idx}')
        print(f"Group nodes: {group['nodes']}")
        print(f"Mean quaternion: {group['q_mean']}")
        print(f"Mean translation: {group['t_mean']}")
        # print(group['imgs'])
    
    
    input()

    for i in range(len(groups)) :
        for j in range(i + 1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            edges = []
            for img1 in group1['nodes']:
                for img2 in group2['nodes']:
                    if img1 in graph and img2 in graph[img1]:
                        edges.append((img1, img2))
            print(f"所有组 {i}-{j} 间边: {edges}")
            # input()
            if (len(edges) < 10):
                # for edge in edges:
                #     delete_edges.append(edge)
                continue
            
            ransac_edges(graph, edges, group1, group2, i, j, max_iter=50)
            # input()
            
    
    # mapping = parse_rgb_file(f'{name}/rgb.txt')

    # # delete_edges = filter_sequential_edges(delete_edges)
    # delete_edges = filter_sequential_edges_1(delete_edges, mapping)

    print("RANSAC 处理完成，删除的边", len(delete_edges))
    input()

    
    # num_edges_after = sum(len(edges) for edges in graph.values())
    # print(f"处理后图中共有 {len(graph)} 个节点，{num_edges_after} 条边")
    
    # with open('modified_view_graph.txt', 'w') as f:
    #     f.write("# Modified view graph after RANSAC processing\n")
    #     for img, edges in graph.items():
    #         f.write(f"{img}: {edges}\n")
    
    with open(f'{name}/delete_edges_name.txt', 'w') as f:
        print("writing delete_edges_name.txt")
        for edge in delete_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    with open(f'{name}/delete_edges.txt', 'w') as f:
        print("writing delete_edges.txt")
        for img1, img2 in delete_edges:
            f.write(f"{img_name2id[img1]} {img_name2id[img2]}\n")
        

    

