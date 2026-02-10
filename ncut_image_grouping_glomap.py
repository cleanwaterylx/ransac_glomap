import numpy as np
from sklearn.cluster import SpectralClustering
import re
import pandas as pd
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from ransac_image_grouping import read_abs_pose_from_glomap, geodesic_angle, geodesic_angle_mat
from ransac_group_glomap_c2w_pnp_2 import colmap_run_feature_extractor, colmap_run_feature_matcher, glomap_run_mapper
import os
import shutil
import networkx as nx
import math
import threading



def parse_image_pair_file(file_path, imgs):
    # 用于存储边信息
    edges = []

    name2idx = {}
    idx2name = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            img1, img2 = parts[2], parts[3]
            inliers = int(parts[4])
            matches = int(parts[5])
            features = int(parts[6])
            # print(f'img1: {img1}, img2: {img2}, inliers: {inliers}, matches: {matches}, features: {features}')
            # input()

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

            edges.append((img1, img2, inliers, matches, features, q, t))

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
    for img1, img2, inliers, matches, features, q, t in edges:
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

    return W, idx2name

def parse_image_pair_file_pose(file_path, imgs):
    # 用于存储边信息
    edges = []

    name2idx = {}
    idx2name = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            img1, img2 = parts[2], parts[3]
            inliers = int(parts[4])
            matches = int(parts[5])
            features = int(parts[6])

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

            edges.append((img1, img2, inliers, matches, features, q, t))

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
    for img1, img2, inliers, matches, features, q, t in edges:
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
        W[i, j] = W[j, i] = a * norm_inliers 
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

    return W, idx2name


def NCut(W, size, idx2name): 
    # 计算拉普拉斯矩阵
    D = np.diag(W.sum(axis=1))
    L = D - W 
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8)) 
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt # 求特征值 
    eigvals, eigvecs = np.linalg.eigh(L_sym) # 排序 
    eigvals = np.sort(eigvals) # 自动检测最大谱间隙位置 
    gaps = np.diff(eigvals) 
    K = int(W.shape[0] / size)
    best_k = np.argmax(gaps[:min(K, len(gaps))]) + 1 # 取前K个特征值防止噪声影响 
    print(f"最佳聚类数（根据谱间隙法）: k = {best_k}") # 使用最佳 k 进行谱聚类 
    sc = SpectralClustering(n_clusters=max(2, best_k), affinity='precomputed', random_state=0) 
    labels = sc.fit_predict(W) 
    
    clusters = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        sub_names = [idx2name[i] for i in indices]
        sub_W = W[np.ix_(indices, indices)]
        print(f"子簇 {label} (大小={len(indices)}): {sub_names}")
        clusters.append(sub_names)

    return clusters


import numpy as np
from sklearn.cluster import SpectralClustering


# 判读group是否一致
def check_from_glomap_result(group_path, cluster):
    image_pair_file_path = os.path.join(group_path, 'sparse/image_pair_inliers_relpose.txt')
    reconstruction_path = os.path.join(group_path, 'sparse/0')
    imgs = read_abs_pose_from_glomap(reconstruction_path)

    # 检查重建结果是否包含足够的图像 90%
    if len(imgs) < int(0.9 * len(cluster)):
        print(f'Warning: Reconstruction has only {len(imgs)} images, expected at least {int(0.9 * len(cluster))}.')
        return False
    
    print('包含足够的图像 90%')

    # 判断相对pose的正确性
    edges = []
    G = nx.Graph()
    with open(image_pair_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            img1, img2 = parts[2], parts[3]
            inliers = int(parts[4])
            matches = int(parts[5])
            features = int(parts[6])
            # print(f'img1: {img1}, img2: {img2}, inliers: {inliers}, matches: {matches}, features: {features}')
            # input()

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

            R_ij, t_ij = R.from_quat(q).as_matrix(), t
            Rel_pose = np.eye(4)
            Rel_pose[:3, :3] = R_ij
            Rel_pose[:3, 3] = t
            edges.append((img1, img2, Rel_pose))
            G.add_edge(img1, img2)

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

            if rR > 20 or rt > 30:
                print(f'Warning: Large error between {img1} and {img2}: rR={rR}, rt={rt}')
                return False

    print('相对pose的正确')

    # 判断回环的正确性
    all_simple_cycles = list(nx.cycle_basis(G))
    Accumulated_pose = np.eye(4)
    for cycle in all_simple_cycles:
        for i in range(len(cycle)):
            img1 = cycle[i]
            img2 = cycle[(i + 1) % len(cycle)]
            # 找到边的相对位姿
            for e in edges:
                if (e[0] == img1 and e[1] == img2):
                    Rel_pose = e[2]
                    break
                if (e[0] == img2 and e[1] == img1):
                    Rel_pose = np.linalg.inv(e[2])
                    break
            else:
                print(f'Error: Edge between {img1} and {img2} not found in edges.')
                return False

            if i == 0:
                Accumulated_pose = Rel_pose
            else:
                Accumulated_pose = Accumulated_pose @ Rel_pose
        
        # translation_error = np.linalg.norm(Accumulated_pose[:3, 3])
        R_acc = Accumulated_pose[:3, :3]
        rotation_error = np.degrees(geodesic_angle_mat(R_acc))
        if rotation_error > 5.0 * math.floor((len(cycle) - 1) / 2) :
            print(f'Warning: Large loop closure error in cycle {cycle}: , rotation_error={rotation_error}')
            return False    
        
    return True          


def run_glomap_for_group(rec_path):
    os.makedirs(rec_path, exist_ok=True)

    db_name = 'database.db'
    db_path = os.path.join(rec_path, db_name)
    image_path = os.path.join(rec_path, 'input')
    output_path = os.path.join(rec_path, 'sparse')

    colmap_run_feature_extractor(db_path, image_path, show_progress=False)
    colmap_run_feature_matcher(db_path, show_progress=False)
    glomap_run_mapper(db_path, image_path, output_path, show_progress=False)



final_clusters = []

def ncut_image_grouping_glomap(name, depth):
    # name = '/home/disk3_SSD/ylx/data/26_ablation'
    # shutil.rmtree(f'{name}/groups', ignore_errors=True)  # 删除已存在的 groups 文件

    imgs = read_abs_pose_from_glomap(f'{name}/sparse/0')

    if depth == 0:
        W1, idx2name = parse_image_pair_file(f'{name}/sparse/image_pair_inliers_relpose_final.txt', imgs)
    elif depth <= 2:
        W1, idx2name = parse_image_pair_file_pose(f'{name}/sparse/image_pair_inliers_relpose_final.txt', imgs)
    else:
        print('Exceeded maximum recursion depth. Stopping further grouping.')
        final_clusters.append((sorted(list(imgs.keys())), name))
        return


    print(len(imgs))

    clusters = NCut(W1, int(20 / (depth + 1)), idx2name)
    threads = []
    for idx, cluster in enumerate(clusters):
        target_input_dir = os.path.join(f'{name}/groups/group_{idx}', 'input')
        os.makedirs(target_input_dir, exist_ok=True)  # 创建目标文件夹

        for img_name in cluster:
            
            src_path = os.path.join(f'{name}/input', img_name)
            dst_path = os.path.join(target_input_dir, img_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)    
                # print(f'Copied {src_path} -> {dst_path}')
            else:
                print(f'Warning: {src_path} does not exist.')
        
        t = threading.Thread(target=run_glomap_for_group, args=(os.path.join(f'{name}/groups/group_{idx}'),))
        print(f'Starting glomap for group {idx} in a new thread.')
        threads.append(t)
        t.start()

    for t in threads:
        t.join()    
    print('All glomap processes completed.')


    for idx, cluster in enumerate(clusters):
        group_path = os.path.join(f'{name}/groups/group_{idx}')
        consistent = check_from_glomap_result(group_path, cluster[0])
        if consistent:
            print(f'Group {idx} is consistent.')
            final_clusters.append(cluster)
        else:
            print(f'Group {idx} is NOT consistent.')
            ncut_image_grouping_glomap(os.path.join(f'{name}/groups/group_{idx}'), depth + 1)


if __name__ == "__main__":
    shutil.rmtree(f'/home/disk3_SSD/ylx/dataset_glg_sfm/cup_test/groups', ignore_errors=True)  # 删除已存在的 groups 文件
    ncut_image_grouping_glomap('/home/disk3_SSD/ylx/dataset_glg_sfm/cup_test', 0)

    print('Final clusters:', final_clusters)


    with open(f'/home/disk3_SSD/ylx/dataset_glg_sfm/cup_test/image_clusters.txt', 'w') as f:
        for idx, (c, name) in enumerate(final_clusters):
            f.write(f'# Cluster {idx}, size: {len(c)}\n')
            f.write(f'Group path: {name}\n')
            for img in c:
                f.write(f'{img} ')
            f.write('\n')

