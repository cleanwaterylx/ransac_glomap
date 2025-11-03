import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def truncate_time(t_str):
    """把时间字符串保留到小数点后两位（不四舍五入，截断）"""
    f = float(t_str)
    truncated = int(f * 100) / 100  # 先放大，再取整，最后缩小
    return f"{truncated:.2f}"

name = 'einstein_1'

rgb_file = f"{name}/rgb.txt"
gt_file = f"{name}/groundtruth.txt"
output_file = f"{name}/img_pose_gt.txt"

# 读 gt.txt
gt_map = {}
with open(gt_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        time_str = truncate_time(parts[0])
        tx, ty, tz = map(float, parts[1:4])  # 只取前三个平移分量
        qx, qy, qz, qw = map(float, parts[4:8])  # 取后四个四元数分量
        gt_map[time_str] = (np.array([qx, qy, qz, qw]), np.array([tx, ty, tz]))

# 读 rgb.txt 并写 img_pose_gt.txt
with open(rgb_file, "r") as f, open(output_file, "w") as out:
    for line in f:
        line = line.strip()
        time_str, img_name = line.split()
        img_name = img_name.replace("rgb/", "")  # 去掉前缀
        time_str = truncate_time(time_str)
        if time_str in gt_map:
            q, t = gt_map[time_str]
            out.write(f"{img_name} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
        else:
            print(f"{time_str} 不在 GT 中, 插值")
            keys = sorted(gt_map.keys())
            # 找到 k1, k2
            for i in range(len(keys)-1):
                if keys[i] <= time_str <= keys[i+1]:
                    k1, k2 = keys[i], keys[i+1]
                    print(f"在 {k1} 和 {k2} 之间插值")
                    break
            else:
                raise ValueError(f"{time_str} 超出范围 [{keys[0]}, {keys[-1]}]")
            
            # 插值
            alpha = (float(time_str) - float(k1)) / (float(k2) - float(k1))
            # 提取位姿
            q1, t1 = gt_map[k1]
            q2, t2 = gt_map[k2]

            # 线性插值平移
            t = (1 - alpha) * np.array(t1) + alpha * np.array(t2)
            # 四元数 SLERP
            rotations = R.from_quat([q1, q2])
            slerp = Slerp([0, 1], rotations)
            q = slerp(alpha).as_quat()

            out.write(f"{img_name} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")


