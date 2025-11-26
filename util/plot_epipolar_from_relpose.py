import math
from cycler import K
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
from matplotlib.widgets import TextBox
from read_write_model import *
from matplotlib import colors as mcolors
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import os

input_model = '/home/disk3_SSD/ylx/data/26_ablation'
click_pos_x = 0
click_pos_y = 0
fig, axs = plt.subplots(1, 2)
fig.suptitle('Epipolar lines')
image_name_1 = ''
image_name_2 = ''
cameras = []
images = []
points3D = []
images_name_id_dic = {}
view_graph_dict = {}

def parse_view_graph(path):
    """
    读取 view_graph.txt，返回 dict[src_filename][dst_filename] = 
      {'rotation_quat': [x,y,z,w], 'translation': [tx,ty,tz]}
    """
    graph = {}

    # 匹配源行：文件名[ID]:
    line_re = re.compile(r'^(?P<src>\S+)\[\d+\]:\s*(?P<body>.+)$')
    # 匹配邻居条目：
    #   - (?P<dst>\S+)
    #   - 四元数：三个带 i/j/k 后缀的分量及最后一个实部，用 +/- 和空格分隔
    #   - 平移：三个浮点数，用空格分隔
    neigh_re = re.compile(
        r'(?P<dst>\S+)\('
        r'(?P<qx>[-0-9.eE+]+)i\s*\+\s*(?P<qy>[-0-9.eE+]+)j\s*\+\s*(?P<qz>[-0-9.eE+]+)k\s*\+\s*(?P<qw>[-0-9.eE+]+)'
        r'\s*,\s*'
        r'(?P<tx>[-0-9.eE+]+)\s+(?P<ty>[-0-9.eE+]+)\s+(?P<tz>[-0-9.eE+]+)'
        r'\)'
    )

    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            m = line_re.match(line)
            if not m:
                continue

            src = m.group('src')
            body = m.group('body')
            graph.setdefault(src, {})

            for nm in neigh_re.finditer(body):
                dst = nm.group('dst')
                # 四元数分量
                qx = float(nm.group('qx'))
                qy = float(nm.group('qy'))
                qz = float(nm.group('qz'))
                qw = float(nm.group('qw'))
                # 平移
                tx = float(nm.group('tx'))
                ty = float(nm.group('ty'))
                tz = float(nm.group('tz'))

                graph[src][dst] = {
                    'quaternion': [qx, qy, qz, qw],
                    'translation': [tx, ty, tz]
                }

    return graph

def plot_epipolar(id_1, id_2, cameras, images, points3D):
    # image_1 = plt.imread(os.path.join(input_model, f'images/{images[id_1].name}'))  
    # image_2 = plt.imread(os.path.join(input_model, f'images/{images[id_2].name}'))
    image_1 = plt.imread(os.path.join(input_model, f'input/{image_name_1}'))  
    image_2 = plt.imread(os.path.join(input_model, f'input/{image_name_2}'))

    feature_points_1 = images[id_1].xys
    feature_points_2 = images[id_2].xys

    # F = get_F_from_two_images(id_1, id_2, cameras, images, points3D)
    R_rel, t_rel, K1, K2 = get_Rt(id_1, id_2, cameras, images, points3D)
    
    e21 = K1 @ -R_rel.T @ t_rel

    num_points = feature_points_1.shape[0]
    
    # pix_h = np.concatenate([feature_points_1, np.ones((num_points, 1))], 1)
    # epiLines = pix_h @ F.T
    # pts = lineToBorderPoints(epiLines, image_2.shape)

    print(axs)
    axs[0].clear()
    axs[1].clear()

    colors = get_n_colors(num_points)
    axs[0].imshow(image_1)
    axs[0].set_autoscale_on(False)
    p = e21[:2]/e21[2]
     # draw the  epipole
    # axs[0].plot(p[0], p[1], marker='o', color='red', markersize=10)    
    
    def on_click(event):
        # print(f"Clicked at: x={event.xdata:.2f}, y={event.ydata:.2f}")
        click_pos_x = event.xdata
        click_pos_y = event.ydata
        # print(image_1.shape)
        print(f"Clicked at: x={click_pos_x:.2f}, y={click_pos_y:.2f}")
        p_h = np.array([click_pos_x, click_pos_y, 1])
        epiLine = np.cross(R_rel.T @ np.linalg.inv(K2) @ p_h,  R_rel.T @ t_rel) @ np.linalg.inv(K1)
        # print(epiLine)
        x = np.linspace(0, image_2.shape[1], 1000)
        k = -epiLine[0]/epiLine[1]
        b = -epiLine[2]/epiLine[1]
        print('k,b', k, b)
        y = k * x + b
        axs[0].plot(x, y)
        axs[1].plot(click_pos_x,click_pos_y,marker='o', color='red', markersize=6)
        
        fig.canvas.draw()
    
    # axs[0].scatter(feature_points_1[:, 0], feature_points_1[:, 1], c = colors)
    axs[1].imshow(image_2)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()    

def get_n_colors(n):
    colors = [mcolors.to_rgba(c)
                  for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    colors = colors * math.ceil(n / len(colors))
    return colors[:n]
  

def get_Rt(id_1, id_2, cameras, images, points3D):    
    
    quaternion = view_graph_dict[image_name_1][image_name_2]['quaternion']
    R_rel = R.from_quat(quaternion).as_matrix()
    t_rel = view_graph_dict[image_name_1][image_name_2]['translation']
    print(R_rel, t_rel)
    
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    
    # K
    print(cameras[images[1].camera_id].params)
    fx, fy, cx, cy, *other = cameras[images[id_1].camera_id].params
    K1 = np.array([
                    [1972, 0, cx],
                    [0, 1972, cy],
                    [0, 0, 1]
                ])
    fx, fy, cx, cy, *other = cameras[images[id_1].camera_id].params
    K2 = np.array([
                    [1972, 0, cx],
                    [0, 1972, cy],
                    [0, 0, 1]
                ])
    return R_rel, t_rel, K1, K2
  
def on_drop_1(event):
    global image_name_1 
    file_path_1 = event.data.strip()
    if file_path_1.endswith(('.png', '.jpg', '.HEIC')):
        try:
            image_name_1 = file_path_1.split('/')[-1]
            print(image_name_1)
        except Exception as e:
            print(f"无法加载图片: {e}")

def on_drop_2(event):
    global image_name_2 
    file_path_2 = event.data.strip()
    if file_path_2.endswith(('.png', '.jpg', '.HEIC')):
        try:
            image_name_2 = file_path_2.split('/')[-1]
            print(image_name_2)
        except Exception as e:
            print(f"无法加载图片: {e}")            

cameras, images, points3D = read_model(
    path=os.path.join(input_model, "sparse/0"), ext=""
)

view_graph_dict = parse_view_graph('/home/disk3_SSD/ylx/data/26_ablation/sparse/view_graph_after_relpose.txt')

print("num_cameras:", len(cameras))
print("num_images:", len(images))
print("num_points3D:", len(points3D))

for id, image in images.items():
    name = image.name
    images_name_id_dic.update({name:id})
# print(images_name_id_dic)


# 创建 Tkinter 窗口
root = TkinterDnD.Tk()
root.title("拖放图片到两个区域")
root.geometry("600x400")

# 创建第一个拖放区域
label1 = tk.Label(root, text="拖放图片到区域 1", bg="lightgreen", width=30, height=10)
label1.pack(pady=20)
label1.drop_target_register(DND_FILES)
label1.dnd_bind('<<Drop>>', on_drop_1)

# 创建第二个拖放区域
label2 = tk.Label(root, text="拖放图片到区域 2", bg="lightblue", width=30, height=10)
label2.pack(pady=20)
label2.drop_target_register(DND_FILES)
label2.dnd_bind('<<Drop>>', on_drop_2)

# 运行 Tkinter 主循环
root.mainloop()


plot_epipolar(int(1), int(2), cameras, images, points3D)

# plt.show()
    
    
    



