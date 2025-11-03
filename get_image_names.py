import os

files = os.listdir('/home/disk3_SSD/ylx/data/4/input')  # 当前目录
files.sort()  # 降序排序
print(' '.join(files))