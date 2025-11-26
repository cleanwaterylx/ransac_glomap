import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

model = 'data/aisle3_4/data'

def read_features(file_path):
    image_features = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(':')
            image_id_info = parts[0].strip()
            data = parts[1].strip().split()
            image_name = data[0]
            coords = list(map(float, data[1:]))
            keypoints = np.array(coords).reshape(-1, 2)
            image_features[image_name] = keypoints
    return image_features

def read_matches(file_path):
    matches = {}
    current_pair = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if ':' in line:
                parts = line.split(':')
                ids = parts[0].strip().split()
                names = parts[1].strip().split()
                image1, image2 = names
                current_pair = (image1, image2)
                matches[current_pair] = []
            else:
                idx1, idx2 = map(int, line.split())
                matches[current_pair].append((idx1, idx2))
    return matches

def draw_matches(img1, kp1, img2, kp2, matches, num_matches=20):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    sampled_matches = random.sample(matches, min(num_matches, len(matches)))
    for idx1, idx2 in sampled_matches:
        pt1 = tuple(np.round(kp1[idx1]).astype(int))
        pt2 = tuple(np.round(kp2[idx2]).astype(int) + np.array([w1, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, pt1, 8, color, -1)
        cv2.circle(canvas, pt2, 8, color, -1)
        cv2.line(canvas, pt1, pt2, color, 4)

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas[..., ::-1])
    plt.axis('off')
    plt.title('Random Feature Matches')
    plt.show()

def visualize_matches(image1_name, image2_name, image_dir,
                      feature_file='sparse/image_feature.txt',
                      match_file='sparse/image_matches.txt'):
    features = read_features(os.path.join(model, feature_file))
    matches = read_matches(os.path.join(model,match_file))
    print('length of matches: ', len(matches))

    if (image1_name, image2_name) not in matches and (image2_name, image1_name) not in matches:
        print("No matches found between the given images.")
        return

    key = (image1_name, image2_name) if (image1_name, image2_name) in matches else (image2_name, image1_name)
    match_pairs = matches[key]
    kp1 = features[image1_name]
    kp2 = features[image2_name]

    if key != (image1_name, image2_name):
        # Swap indices if image order was reversed
        match_pairs = [(b, a) for a, b in match_pairs]

    img1 = cv2.imread(f"{image_dir}/{image1_name}")
    img2 = cv2.imread(f"{image_dir}/{image2_name}")

    draw_matches(img1, kp1, img2, kp2, match_pairs, num_matches=20)

# 示例调用
visualize_matches("00054.png", "00059.png", os.path.join(model, 'input'))

# features = read_features(os.path.join(model, 'sparse/image_feature.txt'))
# matches = read_matches(os.path.join(model, 'sparse/image_matches.txt'))

# count = 0
# for match in matches.items():
#     if len(match[1]) < 50:
#         count = count+1
#         print(match[0], len(match[1]))
# print(len(matches))
# print(count)
# print(matches[('00053.png', '00054.png')])
        

