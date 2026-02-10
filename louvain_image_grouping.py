import numpy as np
import re
import pandas as pd
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from ransac_image_grouping import read_abs_pose_from_glomap, geodesic_angle
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt

img_name2id = {}
name2idx = {}
idx2name = {}

def parse_image_map_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            img_id, image_name = line.split(' ')
            img_name2id.update({image_name.strip(): int(img_id)})

def parse_image_pair_file(file_path):
    # 用于存储边信息
    edges = []
    G = nx.Graph()

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
            G.add_edge(img1, img2, weight=inliers)
    print("Graph info: nodes edges", G.number_of_nodes(), G.number_of_edges())
    return G



if __name__ == '__main__':
    name = '/home/disk3_SSD/ylx/dataset_glg_sfm/cup'
    parse_image_map_file(f'{name}/sparse/image_map.txt')
    # input()
    G = parse_image_pair_file(f'{name}/sparse/image_pair_inliers_relpose.txt')
    communitys = community_louvain.best_partition(G)

    clusters = defaultdict(list)
    for node, cid in communitys.items():
        clusters[cid].append(node)
    clusters = dict(clusters)
    clusters_sorted = sorted(
        clusters.values(),
        key=lambda x: len(x),
        reverse=True
    )

    # use pi3 model to classify
    for idx, cluster in enumerate(clusters_sorted):
        # sample in path in view graph and batched processing
        batch_image_lists = []
        sample_num = 6
        subgraph = G.subgraph(cluster).copy()
        for node, deg in subgraph.degree():
            print(node, deg)
        input()
        # plt.figure(figsize=(8, 8))
        # pos = nx.spring_layout(subgraph, seed=42)
        # nx.draw(subgraph, pos, node_size=80, with_labels=True, font_size=10)
        # plt.show()



    # use pi3 model to classify


    print(clusters_sorted)

    with open(f'{name}/image_clusters_louvain.txt', 'w') as f:
        for idx, c in enumerate(clusters_sorted):
            f.write(f'# Cluster {idx}, size: {len(c)}\n')
            for img in c:
                f.write(f'{img} ')
            f.write('\n')