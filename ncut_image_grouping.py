import numpy as np
from sklearn.cluster import SpectralClustering
import re
import pandas as pd
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from ransac_image_grouping import read_abs_pose_from_glomap, geodesic_angle

img_name2id = {}
name2idx = {}
idx2name = {}

def parse_image_map_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            img_id, image_name = line.split(' ')
            img_name2id.update({image_name.strip(): int(img_id)})

def parse_image_pair_file(file_path, img):
    # 用于存储边信息
    edges = []

    with open(file_path, "r") as f:
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
                raise ValueError(f"无法解析 quaternion: {parts[6:12]}")

            # 去掉 i,j,k
            def clean_token(tok):
                return float(re.sub(r'[ijk]', '', tok))

            qx, qy, qz, qw = map(clean_token, q_tokens)
            # 顺序 (x, y, z, w)
            q = np.array([qx, qy, qz, qw])

            # 平移向量
            tx, ty, tz = map(float, parts[13:16])
            t = np.array([tx, ty, tz])

            edges.append((img1, img2, inliers, inliers / matches, q, t))

    # 提取所有图像名
    image_names = sorted(set([e[0] for e in edges] + [e[1] for e in edges]))

    # 建立索引映射
    for i, name in enumerate(image_names):
        name2idx[name] = i
        idx2name[i] = name

    # 初始化邻接矩阵
    n = len(image_names)
    W = np.zeros((n, n), dtype=float)

    print('len(edges)', len(edges))
    # 填充权重矩阵
    for img1, img2, inliers, inliers_ratio, q, t in edges:
        i, j = name2idx[img1], name2idx[img2]
        R_ij, t_ij = R.from_quat(q).as_matrix(), t
        if img1 in imgs and img2 in imgs:
            R_i, p_i = R.from_quat(imgs[img1]['pose'][0]).as_matrix(), imgs[img1]['pose'][1]
            R_j, p_j = R.from_quat(imgs[img2]['pose'][0]).as_matrix(), imgs[img2]['pose'][1]

            rR = np.degrees(geodesic_angle( R_ij, R_j.T @ R_i))
            d_ij = (p_i - p_j) / np.linalg.norm(p_i - p_j)
            d_ij = R_j.T @ d_ij
            t_ij = t_ij / np.linalg.norm(t_ij)
            cosv =  np.clip(np.dot(d_ij, t_ij), -1.0, 1.0)
            rt = np.degrees(np.arccos(cosv))
        else:
            rR = 180
            rt = 180
        # print(img1, img2)
        # print(rR, rt)
        
        sigma_r = 5.0
        sigma_t = 10.0
        s_r = np.exp(- (rR / float(sigma_r))**2)
        s_t = np.exp(- (rt / float(sigma_t))**2)

        # weight = a * norm_inliers + b * (1 - rR / 180.0) + c * (1 - rt / 180.0)
        # weight = a * inliers_ratio + b * s_r + c * s_t
        # weight = max(weight, 0)
        # W[i, j] = W[j, i] = weight  # 无向图对称
        W[i, j] = W[j, i] = inliers
        # W[i, j] = W[j, i] = inliers * s_r * s_t  # 无向图对称
        # print(rR, rt)
        # print(f'inliers: {inliers}, norm_inliers: {norm_inliers:.4f}, s_r: {s_r:.4f}, s_t: {s_t:.4f} => weight: {weight:.4f}')
        # print(inliers_ratio, s_r, s_t)
        # input()

    # 打印结果
    # print("图像名索引：")
    # for name, idx in name2idx.items():
    #     print(f"{idx}: {name}")

    print("\n权重矩阵：")
    df = pd.DataFrame(W, index=image_names, columns=image_names)
    print(df)

    return W

def parse_image_pair_file_pose(file_path, img):
    # 用于存储边信息
    edges = []

    with open(file_path, "r") as f:
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
                raise ValueError(f"无法解析 quaternion: {parts[6:12]}")

            # 去掉 i,j,k
            def clean_token(tok):
                return float(re.sub(r'[ijk]', '', tok))

            qx, qy, qz, qw = map(clean_token, q_tokens)
            # 顺序 (x, y, z, w)
            q = np.array([qx, qy, qz, qw])

            # 平移向量
            tx, ty, tz = map(float, parts[13:16])
            t = np.array([tx, ty, tz])

            edges.append((img1, img2, inliers, inliers / matches, q, t))

    # 提取所有图像名
    image_names = sorted(set([e[0] for e in edges] + [e[1] for e in edges]))

    # 建立索引映射
    for i, name in enumerate(image_names):
        name2idx[name] = i
        idx2name[i] = name

    # 初始化邻接矩阵
    n = len(image_names)
    W = np.zeros((n, n), dtype=float)

    print('len(edges)', len(edges))
    max_inliers = max([e[2] for e in edges])
    a, b, c = 0.8, 0.1, 0.1

    # 填充权重矩阵
    for img1, img2, inliers, inliers_ratio, q, t in edges:
        i, j = name2idx[img1], name2idx[img2]
        R_ij, t_ij = R.from_quat(q).as_matrix(), t
        if img1 in imgs and img2 in imgs:
            R_i, p_i = R.from_quat(imgs[img1]['pose'][0]).as_matrix(), imgs[img1]['pose'][1]
            R_j, p_j = R.from_quat(imgs[img2]['pose'][0]).as_matrix(), imgs[img2]['pose'][1]

            rR = np.degrees(geodesic_angle( R_ij, R_j.T @ R_i))
            d_ij = (p_i - p_j) / np.linalg.norm(p_i - p_j)
            d_ij = R_j.T @ d_ij
            t_ij = t_ij / np.linalg.norm(t_ij)
            cosv =  np.clip(np.dot(d_ij, t_ij), -1.0, 1.0)
            rt = np.degrees(np.arccos(cosv))
        else:
            rR = 180
            rt = 180
        # print(img1, img2)
        # print(rR, rt)
        
        sigma_r = 5.0
        sigma_t = 10.0
        s_r = np.exp(- (rR / float(sigma_r))**2)
        s_t = np.exp(- (rt / float(sigma_t))**2)

        norm_inliers = inliers / max_inliers
        # weight = a * norm_inliers + b * (1 - rR / 180.0) + c * (1 - rt / 180.0)
        # weight = a * inliers_ratio + b * s_r + c * s_t
        # weight = max(weight, 0)
        # W[i, j] = W[j, i] = weight  # 无向图对称
        # W[i, j] = W[j, i] = inliers_ratio * norm_inliers * 100  + (1 - inliers_ratio) /2 * (1 - rR / 180.0) + (1 - inliers_ratio) /2 * (1 - rt / 180.0)
        W[i, j] = W[j, i] = norm_inliers * 100  + (1 - rR / 180.0) + (1 - rt / 180.0)
        # print(inliers_ratio * norm_inliers * 100, (1 - inliers_ratio) /2 * (1 - rR / 180.0) + (1 - inliers_ratio) /2 * (1 - rt / 180.0))
        # input()
        # W[i, j] = W[j, i] = inliers * s_r * s_t  # 无向图对称
        # print(rR, rt)
        # print(f'inliers: {inliers}, norm_inliers: {norm_inliers:.4f}, s_r: {s_r:.4f}, s_t: {s_t:.4f} => weight: {weight:.4f}')
        # print(inliers_ratio, s_r, s_t)
        # input()

    # 打印结果
    # print("图像名索引：")
    # for name, idx in name2idx.items():
    #     print(f"{idx}: {name}")

    print("\n权重矩阵：")
    df = pd.DataFrame(W, index=image_names, columns=image_names)
    print(df)

    return W


import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering

def recursive_NCut(W, idx2name, max_group_size=15, min_group_size=7, depth=0):
    """
    对权重矩阵 W 执行递归谱聚类分割，使每个子集节点数在 [min_group_size, max_group_size] 内。
    """
    n = W.shape[0]
    indent = "  " * depth

    # 若当前组大小已经在合适范围内，停止分割
    if n <= max_group_size:
        print(f"{indent}组大小={n}，停止分割。")
        return [list(idx2name.values())]

    # --- 计算对称归一化拉普拉斯矩阵 ---
    D = np.diag(W.sum(axis=1))
    L = D - W
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # --- 求特征值 ---
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    eigvals = np.sort(eigvals)

    # --- 自动选择最佳 k ---
    gaps = np.diff(eigvals)
    best_k = np.argmax(gaps[:min(10, len(gaps))]) + 1
    best_k = max(2, min(best_k, n))  # 至少分两类
    print(f"{indent}检测到最佳聚类数 k = {best_k}")

    # --- 执行谱聚类 ---
    sc = SpectralClustering(n_clusters=best_k, affinity='precomputed', random_state=0)
    labels = sc.fit_predict(W)

    clusters = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        sub_names = [idx2name[i] for i in indices]
        print(f"{indent}子簇 {label} (大小={len(indices)}): {sub_names}")

        # 如果簇太小，不进行递归
        if len(indices) < min_group_size:
            print(f"{indent}⚠️ 子簇过小 (size={len(indices)} < {min_group_size})，跳过该子簇。")
            return [list(idx2name.values())] 

        # 若子簇仍大于 max_group_size，则递归细分
        if len(indices) > max_group_size:
            sub_W = W[np.ix_(indices, indices)]
            sub_idx2name = {i: idx2name[indices[i]] for i in range(len(indices))}
            clusters.extend(recursive_NCut(sub_W, sub_idx2name, max_group_size, min_group_size, depth + 1))
        else:
            clusters.append(sub_names)

    # 若没有任何有效分簇（都太小），返回原簇
    if not clusters:
        print(f"{indent}所有子簇过小，保留原簇。")
        return [list(idx2name.values())]

    return clusters

# def recursive_NCut(W1, W2, idx2name, max_group_size=10, min_group_size=5, depth=0, use_W2=False):
#     """
#     对权重矩阵执行递归谱聚类分割：
#     - 第一次调用使用 W1（内点数矩阵）；
#     - 子簇若需继续分割，则改用 W2。
#     """
#     indent = "  " * depth
#     n = len(idx2name)

#     # 若当前组大小已经在合适范围内，停止分割
#     if n <= max_group_size:
#         print(f"{indent}组大小={n}，停止分割。")
#         return [list(idx2name.values())]

#     # 当前层使用哪个矩阵？
#     W = W2 if use_W2 else W1
#     print(f"{indent}使用 {'W2' if use_W2 else 'W1'} 进行谱聚类 (n={n})")

#     # --- 计算对称归一化拉普拉斯矩阵 ---
#     D = np.diag(W.sum(axis=1))
#     L = D - W
#     D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
#     L_sym = D_inv_sqrt @ L @ D_inv_sqrt

#     # --- 求特征值 ---
#     eigvals, eigvecs = np.linalg.eigh(L_sym)
#     eigvals = np.sort(eigvals)

#     # --- 自动选择最佳 k ---
#     gaps = np.diff(eigvals)
#     best_k = np.argmax(gaps[:10]) + 1
#     best_k = max(2, min(best_k, n))
#     print(f"{indent}检测到最佳聚类数 k = {best_k}")

#     # --- 执行谱聚类 ---
#     sc = SpectralClustering(n_clusters=best_k, affinity='precomputed', random_state=0)
#     labels = sc.fit_predict(W)

#     clusters = []
#     all_too_small = True

#     for label in np.unique(labels):
#         indices = np.where(labels == label)[0]
#         sub_names = [idx2name[i] for i in indices]
#         print(f"{indent}子簇 {label} (大小={len(indices)}): {sub_names}")

#         if len(indices) < min_group_size:
#             print(f"{indent}⚠️ 子簇过小 (size={len(indices)} < {min_group_size})，保留原簇不分割。")
#             return [list(idx2name.values())]
#         else:
#             all_too_small = False

#         # 若子簇仍大于 max_group_size，则递归细分，改用 W2
#         if len(indices) > max_group_size:
#             sub_W1 = W1[np.ix_(indices, indices)]
#             sub_W2 = W2[np.ix_(indices, indices)]
#             sub_idx2name = {i: idx2name[indices[i]] for i in range(len(indices))}

#             # 下一层改用 W2
#             clusters.extend(
#                 recursive_NCut(sub_W1, sub_W2, sub_idx2name,
#                                max_group_size, min_group_size,
#                                depth + 1, use_W2=True)
#             )
#         else:
#             clusters.append(sub_names)

#     if all_too_small:
#         print(f"{indent}所有分组都太小，放弃分割。")
#         return [list(idx2name.values())]

#     return clusters

def NCut(W): 
    # 计算拉普拉斯矩阵
    D = np.diag(W.sum(axis=1))
    L = D - W 
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8)) 
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt # 求特征值 
    eigvals, eigvecs = np.linalg.eigh(L_sym) # 排序 
    eigvals = np.sort(eigvals) # 自动检测最大谱间隙位置 
    gaps = np.diff(eigvals) 
    best_k = np.argmax(gaps[:10]) + 1 # 取前10个特征值防止噪声影响 
    best_k = int(W.shape[0] / 20)
    # print(f"最佳聚类数（根据谱间隙法）: k = {best_k}") # 使用最佳 k 进行谱聚类 
    sc = SpectralClustering(n_clusters=best_k, affinity='precomputed', random_state=0) 
    labels = sc.fit_predict(W) 
    
    clusters = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        sub_names = [idx2name[i] for i in indices]
        print(f"子簇 {label} (大小={len(indices)}): {sub_names}")
        clusters.append(sub_names)

    return clusters




if __name__ == '__main__':
    name = '/home/disk3_SSD/ylx/data/37'
    parse_image_map_file(f'{name}/sparse/image_map.txt')
    imgs = read_abs_pose_from_glomap(f'{name}/sparse/0')
    # input()
    W1 = parse_image_pair_file(f'{name}/sparse/image_pair_inliers_relpose.txt', imgs)
    # W2 = parse_image_pair_file_pose(f'{name}/sparse/image_pair_inliers_relpose.txt', imgs)
    # print(name2idx)
    # print(idx2name)

    print(len(imgs))

    # clusters = recursive_NCut(W, idx2name)
    # clusters = recursive_NCut(W1, idx2name)
    # clusters = NCut(W1)
    # with open(f'{name}/image_clusters.txt', 'w') as f:
    #     for idx, c in enumerate(clusters):
    #         f.write(f'# Cluster {idx}, size: {len(c)}\n')
    #         for img in c:
    #             f.write(f'{img} ')
    #         f.write('\n')

    print("----- Using pose-based weights -----")

    clusters = recursive_NCut(W1, idx2name)
    # clusters = NCut(W1)
    with open(f'{name}/image_clusters.txt', 'w') as f:
        for idx, c in enumerate(clusters):
            f.write(f'# Cluster {idx}, size: {len(c)}\n')
            for img in c:
                f.write(f'{img} ')
            f.write('\n')
