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

input_model = 'data/aisle3/data'
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

def plot_epipolar(image_name1, image_name2):
    # image_1 = plt.imread(os.path.join(input_model, f'images/{images[id_1].name}'))  
    # image_2 = plt.imread(os.path.join(input_model, f'images/{images[id_2].name}'))
    image_1 = plt.imread(os.path.join(input_model, f'input/{image_name1}'))  
    image_2 = plt.imread(os.path.join(input_model, f'input/{image_name2}'))


    # F = get_F_from_two_images(id_1, id_2, cameras, images, points3D)
    R_rel, t_rel, K1, K2 = get_Rt()
    
    e21 = K1 @ -R_rel.T @ t_rel
    
    # pix_h = np.concatenate([feature_points_1, np.ones((num_points, 1))], 1)
    # epiLines = pix_h @ F.T
    # pts = lineToBorderPoints(epiLines, image_2.shape)

    print(axs)
    axs[0].clear()
    axs[1].clear()

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
  

def get_Rt():    

    quaternion1 = np.array([-0.00468834, -0.00878352, -0.00319051, -0.99994534])
    R1 = R.from_quat(quaternion1).as_matrix()
    t1 = np.array([ 4.88854369, -4.52025921, 29.11247936])
    
    quaternion2 = np.array([-0.01276474, -0.04386656, -0.0092918 ,  0.99891263])
    R2 = R.from_quat(quaternion2).as_matrix()
    t2 = np.array([ 1.66561892, -6.01068276, 31.94754054])

    
    R_rel = R2.T @ R1  # 计算相对旋转
    t_rel = R2.T @ (t1 - t2)
    
    
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    
    # K
    K1 = np.array([
                    [856.2, 0, 540],
                    [0, 856.8, 960],
                    [0, 0, 1]
                ])

    K2 = np.array([
                    [874, 0, 540],
                    [0, 843, 960],
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

plot_epipolar(image_name_1, image_name_2)
# plt.show()
    
    
    



