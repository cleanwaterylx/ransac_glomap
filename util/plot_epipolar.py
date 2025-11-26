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

input_model = '/home/disk3_SSD/ylx/data/26_ablation/'
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
        

def plot_epipolar(id_1, id_2, cameras, images, points3D):
    # image_1 = plt.imread(os.path.join(input_model, f'images/{images[id_1].name}'))  
    # image_2 = plt.imread(os.path.join(input_model, f'images/{images[id_2].name}'))
    image_1 = plt.imread(os.path.join(input_model, f'input/{images[id_1].name}'))  
    image_2 = plt.imread(os.path.join(input_model, f'input/{images[id_2].name}'))

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
    print(images[id_1])
    print(images[id_2])
    
    quaternion = images[id_1].qvec
    quaternion = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    R1 = R.from_quat(quaternion).as_matrix()
    t1 = images[id_1].tvec
    
    quaternion = images[id_2].qvec
    quaternion = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    R2 = R.from_quat(quaternion).as_matrix()
    t2 = images[id_2].tvec

    
    R_rel = R2 @ R1.T  # 计算相对旋转
    t_rel = t2 - R_rel @ t1
    # print("R_rel:", R_rel)
    # print("t_rel:", t_rel)
    
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    
    # K
    print(cameras[images[id_1].camera_id].params)
    fx, fy, cx, cy, *other = cameras[images[id_1].camera_id].params
    K1 = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
    fx, fy, cx, cy, *other = cameras[images[id_2].camera_id].params
    K2 = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
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

print("num_cameras:", len(cameras))
print("num_images:", len(images))
print("num_points3D:", len(points3D))

for id, image in images.items():
    name = image.name
    print(name)
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

id1 = -1
id2 = -1
if image_name_1 in images_name_id_dic:
    id1 = images_name_id_dic[image_name_1]
else:
    print("Invalid Input 1")

if image_name_2 in images_name_id_dic:
    id2 = images_name_id_dic[image_name_2]
else:
    print("Invalid Input 2")
plot_epipolar(int(id1), int(id2), cameras, images, points3D)

# plt.show()
    
    
    



