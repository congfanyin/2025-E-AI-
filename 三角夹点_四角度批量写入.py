import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import statistics as st
import re

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = "附件/附件1/运动者1的跳远位置信息_修正.xlsx"
post_folder = r"D:\math\附件\附件1"
output_file = "跳远角度汇总.xlsx"


# -------------------------------
# 读取 Excel 文件
# -------------------------------
def read_excel_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    return all_files  # 只返回路径


# -------------------------------
# 三点夹角计算函数
# -------------------------------
def angle_between_three_points(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# -------------------------------
# 倾斜角度计算函数（肩-髋连线 vs 竖直方向）
# -------------------------------
def tilt_angle(A, B):
    vec = np.array(A) - np.array(B)
    vertical = np.array([0, -1])
    cosine_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# -------------------------------
# 从文件名提取信息
# -------------------------------
def parse_filename(filename):
    # 名字：运动者编号
    name_match = re.search(r'运动者\d+', filename)
    name = name_match.group() if name_match else "未知"

    # 调整：前/后
    adjust = "前" if "调整前" in filename or "第" in filename and "调整后" not in filename else "后"

    # 次数：第几次
    times_match = re.search(r'第(\d+)次', filename)
    times = f"第{times_match.group(1)}次" if times_match else "未知"

    return name, adjust, times


# -------------------------------
# 计算文件角度并整理成字典
# -------------------------------
def compute_file_metrics(file_path):
    df = pd.read_excel(file_path)
    results = {}

    # 第一行：下蹲关节角度
    row1 = df.iloc[0]
    hka = angle_between_three_points(
        (st.mean([row1['24_X'], row1['23_X']]), st.mean([row1['24_Y'], row1['23_Y']])),
        (st.mean([row1['26_X'], row1['25_X']]), st.mean([row1['26_Y'], row1['25_Y']])),
        (st.mean([row1['28_X'], row1['27_X']]), st.mean([row1['28_Y'], row1['27_Y']]))
    )
    shk = angle_between_three_points(
        (st.mean([row1['11_X'], row1['12_X']]), st.mean([row1['11_Y'], row1['12_Y']])),
        (st.mean([row1['23_X'], row1['24_X']]), st.mean([row1['23_Y'], row1['24_Y']])),
        (st.mean([row1['25_X'], row1['26_X']]), st.mean([row1['25_Y'], row1['26_Y']]))
    )
    jhx = angle_between_three_points(
        (st.mean([row1['25_X'], row1['26_X']]), st.mean([row1['25_Y'], row1['26_Y']])),
        (st.mean([row1['27_X'], row1['28_X']]), st.mean([row1['27_Y'], row1['28_Y']])),
        (st.mean([row1['31_X'], row1['32_X']]), st.mean([row1['31_Y'], row1['32_Y']]))

    )
    wrist = angle_between_three_points(
        (st.mean([row1['19_X'], row1['20_X']]), st.mean([row1['19_Y'], row1['20_Y']])),
        (st.mean([row1['11_X'], row1['12_X']]), st.mean([row1['11_Y'], row1['12_Y']])),
        (st.mean([row1['23_X'], row1['24_X']]), st.mean([row1['23_Y'], row1['24_Y']]))
    )
    results['发力膝角'] = round(hka, 2)
    results['发力髋角'] = round(shk, 2)
    results['发力踝角'] = round(jhx, 2)
    results['发力肩角'] = round(wrist, 2)

    # 第二行：躯干倾斜
    row1 = df.iloc[0]
    hka = angle_between_three_points(
        (st.mean([row1['24_X'], row1['23_X']]), st.mean([row1['24_Y'], row1['23_Y']])),
        (st.mean([row1['26_X'], row1['25_X']]), st.mean([row1['26_Y'], row1['25_Y']])),
        (st.mean([row1['28_X'], row1['27_X']]), st.mean([row1['28_Y'], row1['27_Y']]))
    )
    shk = angle_between_three_points(
        (st.mean([row1['11_X'], row1['12_X']]), st.mean([row1['11_Y'], row1['12_Y']])),
        (st.mean([row1['23_X'], row1['24_X']]), st.mean([row1['23_Y'], row1['24_Y']])),
        (st.mean([row1['25_X'], row1['26_X']]), st.mean([row1['25_Y'], row1['26_Y']]))
    )
    jhx = angle_between_three_points(
        (st.mean([row1['25_X'], row1['26_X']]), st.mean([row1['25_Y'], row1['26_Y']])),
        (st.mean([row1['27_X'], row1['28_X']]), st.mean([row1['27_Y'], row1['28_Y']])),
        (st.mean([row1['31_X'], row1['32_X']]), st.mean([row1['31_Y'], row1['32_Y']]))

    )
    wrist = angle_between_three_points(
        (st.mean([row1['19_X'], row1['20_X']]), st.mean([row1['19_Y'], row1['20_Y']])),
        (st.mean([row1['11_X'], row1['12_X']]), st.mean([row1['11_Y'], row1['12_Y']])),
        (st.mean([row1['23_X'], row1['24_X']]), st.mean([row1['23_Y'], row1['24_Y']]))
    )
    results['起跳膝角'] = round(hka, 2)
    results['起跳髋角'] = round(shk, 2)
    results['起跳踝角'] = round(jhx, 2)
    results['起跳肩角'] = round(wrist, 2)

    # 文件信息
    name, adjust, times = parse_filename(file_path)
    results['名字'] = name
    results['调整'] = adjust
    results['次数'] = times

    return results


# -------------------------------
# 遍历所有文件
# -------------------------------
all_files = read_excel_files(pre_folder) + read_excel_files(post_folder)
all_data = [compute_file_metrics(f) for f in all_files]

# -------------------------------
# 保存到 Excel
# -------------------------------
df_out = pd.DataFrame(all_data)
# 调整列顺序
cols_order = ['名字', '调整', '次数', '发力膝角', '发力髋角', '发力踝脚', '发力肩角', '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角']
df_out = df_out[cols_order]
df_out.to_excel(output_file, index=False)

print(f"所有文件数据已保存到: {output_file}")
