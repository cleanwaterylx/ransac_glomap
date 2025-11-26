import os
import shutil

def check_consistency(group_folder):
    """
    检查group_folder下的glomap结果是否一致
    返回 True 一致, False 不一致
    """
    # TODO: 根据你的规则实现一致性检查
    return False

def run_glomap(input_folder, output_folder):
    """
    对 input_folder 运行 glomap，结果放到 output_folder
    """
    os.makedirs(output_folder, exist_ok=True)
    # TODO: 在这里调用 glomap
    print(f"Running glomap on {input_folder}, output -> {output_folder}")

def split_group(input_folder):
    """
    根据你的分组逻辑对 input_folder 分组
    返回列表，每个元素是子分组的文件夹路径
    """
    # TODO: 根据实际逻辑分组
    subgroups = [os.path.join(input_folder, f"subgroup_{i}") for i in range(2)]
    for sg in subgroups:
        os.makedirs(sg, exist_ok=True)
    return subgroups

def process_group(input_folder, depth=0):
    """
    递归处理分组
    """
    indent = "  " * depth
    print(f"{indent}Processing group: {input_folder}")

    # 运行 glomap
    output_folder = input_folder + "_glomap"
    run_glomap(input_folder, output_folder)

    # 检查一致性
    if check_consistency(output_folder):
        print(f"{indent}Group is consistent: {input_folder}")
        return
    else:
        print(f"{indent}Group is NOT consistent, splitting...")
        subgroups = split_group(input_folder)
        for sg in subgroups:
            process_group(sg, depth + 1)  # 递归处理子分组

# --------- 主程序 ---------
root_folder = "data"
process_group(root_folder)
