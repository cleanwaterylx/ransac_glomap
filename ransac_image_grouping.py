import numpy as np
import pycolmap
import os
from scipy import cluster
from scipy.spatial.transform import Rotation as R
import re
from ransac_group_glomap_c2w_pnp_1 import invert_pose

imgs = {}

def read_relpose_file(filename):
    edges = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            img1, img2 = parts[2], parts[3]
            inliers = int(parts[4])
            matches = int(parts[5])

            # quaternion 部分在 parts[5:11]，里面有 "+"，需要过滤
            q_tokens = [p for p in parts[6:13] if p != '+']
            if len(q_tokens) != 4:
                raise ValueError(f"无法解析 quaternion: {parts[6:13]}")

            # 去掉 i,j,k
            def clean_token(tok):
                return float(re.sub(r'[ijk]', '', tok))

            qx, qy, qz, qw = map(clean_token, q_tokens)
            # 顺序 (x, y, z, w)
            q = np.array([qx, qy, qz, qw])
            # todo  parts[5:12] 需要修改
            # print(qx, qy, qz, qw)
            # input()

            # 平移向量
            tx, ty, tz = map(float, parts[13:16])
            t = np.array([tx, ty, tz])

            if img1 < img2:
                edges[(img1, img2)] = (q, t)
            else:
                # print(f'Inverting pose for {img2} -> {img1}')
                edges[(img2, img1)] = invert_pose(q, t)

    return edges

def invert_pose(q, t):
    """反转一个位姿：将 q, t 从 A->B 变成 B->A"""
    r_inv = R.from_quat(q).inv()
    t_inv = -r_inv.apply(t)
    return r_inv.as_quat(), t_inv

import numpy as np

def geodesic_angle(R1, R2):
    """
    计算两个旋转矩阵之间的测地距离（弧度）
    R1, R2 : 3x3 numpy 数组
    """
    R = R1.T @ R2
    trace_val = np.trace(R)
    cos_theta = (trace_val - 1) / 2
    # 数值稳定性裁剪
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def geodesic_angle_mat(R_mat):
    """
    计算旋转矩阵的测地距离（弧度）
    R_mat : 3x3 numpy 数组
    """
    trace_val = np.trace(R_mat)
    cos_theta = (trace_val - 1) / 2
    # 数值稳定性裁剪
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def MAD(values):
    """
    计算中位数绝对偏差 (Median Absolute Deviation)
    values: list 或 numpy 数组
    """
    values = np.asarray(values)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return mad

def median(data):
    """
    使用 numpy 计算中位数
    data: list 或 numpy 数组
    """
    return np.median(np.asarray(data))

from collections import defaultdict

def connected_components1(V, edges):
    """
    V: list 或 set，顶点集合
    edges: list of (i, j)，无向边
    返回: list of sets，每个 set 是一个连通分量
    """
    # 构建邻接表
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    clusters = []

    for node in V:
        if node not in visited:
            stack = [node]
            comp = set()
            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    comp.add(cur)
                    stack.extend(graph[cur])
            clusters.append(comp)
    return clusters

def connected_components(edges):
    """
    edges: list of (i, j)，无向边
    返回: list of sets，每个 set 是一个连通分量（点集合）
    """
    if not edges:
        return []

    # 构建邻接表
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    clusters = []

    # 遍历所有点
    for u, v in edges:
        for node in (u, v):
            if node not in visited:
                stack = [node]
                comp = set()
                while stack:
                    cur = stack.pop()
                    if cur not in visited:
                        visited.add(cur)
                        comp.add(cur)
                        stack.extend(graph[cur])
                clusters.append(comp)

    return clusters

def read_abs_pose_from_glomap(rec_path):
    imgs = {}

    reconstruction = pycolmap.Reconstruction(rec_path)
    print(reconstruction.summary())

    for img_id, img in reconstruction.images.items():
        # print(img_id, img.name, img.cam_from_world().rotation.quat, img.cam_from_world().translation)
        
        # convert to c2w
        imgs.update({img.name : {'pose' : invert_pose(img.cam_from_world().rotation.quat, img.cam_from_world().translation)}})
        # print(invert_pose(img.cam_from_world.rotation.quat, img.cam_from_world.translation))

    return imgs

if __name__ == '__main__':

    name = 'aisle3'

    view_graph_file_path = f'/home/disk3_SSD/ylx/data/{name}/sparse/view_graph_after_relpose.txt'
    rec_path = f'/home/disk3_SSD/ylx/data/{name}'

    imgs = read_abs_pose_from_glomap(rec_path)
    edges = read_relpose_file(f'/home/disk3_SSD/ylx/data/{name}/sparse/image_pair_inliers_relpose.txt')
    # print(edges)
    # input()

    # for img1 in graph:
    #     for img2 in graph[img1]:
    #         key = tuple(sorted((img1, img2)))  # 规范化 key
    #         if key not in edges:              # 只保留第一次
    #             edges[key] = graph[img1][img2]
    E_strong = []
    E_bad = []

    imgs_list = sorted(list(imgs.keys()))
    sequential_edges = {}
    for i in range(len(imgs_list)-1):
        img1 = imgs_list[i]
        img2 = imgs_list[i+1]
        if (img1, img2) in edges:
            sequential_edges[(img1, img2)] = edges[(img1, img2)]
        else:
            E_bad.append((img1, img2))

    print(sequential_edges)
    input()


    rR_all, rt_all = [], []

    for (i, j) in sequential_edges:
        R_ij, t_ij = R.from_quat(sequential_edges[(i, j)][0]).as_matrix(), sequential_edges[(i, j)][1]
        R_i, p_i = R.from_quat(imgs[i]['pose'][0]).as_matrix(), imgs[i]['pose'][1]
        R_j, p_j = R.from_quat(imgs[j]['pose'][0]).as_matrix(), imgs[j]['pose'][1]

        # print(R_i, p_i)
        # print(R_j, p_j)
        
        rR = geodesic_angle( R_ij, R_j.T @ R_i) * 180 / np.pi
        d_ij = (p_i - p_j) / np.linalg.norm(p_i - p_j)
        d_ij = R_j.T @ d_ij
        t_ij = t_ij / np.linalg.norm(t_ij)
        rt = np.arccos(abs(np.dot(d_ij, t_ij))) * 180 / np.pi

        # print(i, j)
        # print('R_ij t_ij', R_ij, t_ij)
        # print('q', sequential_edges[(i, j)][0])
        # print('angle', geodesic_angle_mat(R_ij))
        # print(R_j.T @ R_i)
        # print(d_ij, t_ij)

        print(i, j, f'角度误差: {rR:.3f}, 平移误差: {rt:.3f}')
        rR_all.append(rR)
        rt_all.append(rt)
        # input()
    
    sigma_R = MAD(rR_all)
    sigma_t = MAD(rt_all)
    print(f'旋转误差的MAD: {sigma_R:.3f}, 平移误差的MAD: {sigma_t:.3f}')
    # input()

    s_all = []
    wR = 1
    wt = 1

    for (i, j), rR, rt in zip(sequential_edges, rR_all, rt_all):
        r_tilde = wR * rR/sigma_R + wt * rt/sigma_t
        s_all.append(r_tilde)

    tau = median(s_all) + 10 * MAD(s_all)
    print(f'阈值 tau: {tau:.3f}')

    for (i, j), s in zip(sequential_edges, s_all):
        print(i, j, s, tau)  
        if s < tau:
            E_strong.append((i, j))
        else:
            E_bad.append((i, j))
    
    print(len(sequential_edges))
    print(f'强边数量: {len(E_strong)}')
    # print(E_strong)
    # input()

    V = [img for img in imgs]

    cluster_strong = connected_components(E_strong)
    cluster_strong = [list(c) for c in cluster_strong]
    # 按照大小从大到小排序
    cluster_strong.sort(key=len, reverse=True)

    for idx, c in enumerate(cluster_strong):
        c.sort()
        print(f'Cluster {idx}, size: {len(c)}')
        print(c)

    cluster_bad = connected_components(E_bad)
    cluster_bad = [list(c) for c in cluster_bad]
    # 按照大小从大到小排序
    cluster_bad.sort(key=len, reverse=True)

    for idx, c in enumerate(cluster_bad):
        c.sort()
        print(f'Cluster {idx}, size: {len(c)}')
        print(c)

    cluster = cluster_strong + cluster_bad
    cluster.sort(key=len, reverse=True)

    # 保存结果
    with open(f'{name}/image_clusters.txt', 'w') as f:
        for idx, c in enumerate(cluster):
            c.sort()
            f.write(f'# Cluster {idx}, size: {len(c)}\n')
            for img in c:
                f.write(f'{img} ')
            f.write('\n')

    
    

    



