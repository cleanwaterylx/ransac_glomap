import os

files = os.listdir('/home/disk3_SSD/ylx/dataset_glg_sfm/cup/input')  # 当前目录
files.sort()  # 降序排序
print(' '.join(files))