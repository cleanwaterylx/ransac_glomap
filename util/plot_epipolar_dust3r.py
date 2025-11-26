import math
from cycler import K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from read_write_model import *
from matplotlib import colors as mcolors
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import cv2

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
        

def plot_epipolar(id_1, id_2, image_name_1, image_name_2, intrinsics, poses):
    image_1 = plt.imread(os.path.join(input_model, f'input/{image_name_1}'))  
    image_2 = plt.imread(os.path.join(input_model, f'input/{image_name_2}'))
    image_1 = cv2.resize(image_1, (288, 512), interpolation=cv2.INTER_AREA)
    image_2 = cv2.resize(image_2, (288, 512), interpolation=cv2.INTER_AREA)

    # F = get_F_from_two_images(id_1, id_2, cameras, images, points3D)
    R_rel, t_rel, K1, K2 = get_Rt(id_1, id_2, intrinsics, poses)
    print(R_rel, t_rel, K1, K2)
    
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

def get_Rt(id_1, id_2, intrinsics, poses):    

    R1 = poses[id_1][:3, :3]
    t1 = poses[id_1][:3, 3]
    
    R2 = poses[id_2][:3, :3]
    t2 = poses[id_2][:3, 3]

    
    R_rel = R2 @ R1.T  # 计算相对旋转
    t_rel = t2 - R_rel @ t1
    # print("R_rel:", R_rel)
    # print("t_rel:", t_rel)
    
    # K
    K1 = intrinsics[0]
    K2 = intrinsics[1]
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


intrinsics = np.load(input_model + '/intrinsics.npy')
poses = np.load(input_model + '/poses.npy')
images = np.load(input_model + '/images.npy')

# print("num_cameras:", len(cameras))
# print("num_images:", len(images))
# print("num_points3D:", len(points3D))

# for i in range(0, 112):
#     name = '{:05d}'.format(i) + '.png'
#     images_name_id_dic.update({name:i})
# print(images_name_id_dic)
for index, filename in enumerate(images):
    images_name_id_dic[filename] = index
print(images_name_id_dic)


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

if image_name_1 in images_name_id_dic and image_name_2 in images_name_id_dic:
    id1,id2 = images_name_id_dic[image_name_1], images_name_id_dic[image_name_2]
    print(id1, id2)
    plot_epipolar(int(id1), int(id2), image_name_1, image_name_2, intrinsics, poses)
else:
    print("Invalid Input")
# plot_epipolar(1, 1, image_name_1, image_name_2, intrinsics, poses)

# plt.show()
    
    
    



