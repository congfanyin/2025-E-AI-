import os
import glob
import pandas as pd
import numpy as np
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
    if os.path.isfile(folder_path):
        return [folder_path]  # 如果传的是文件路径
    all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    return all_files


# -------------------------------
# 三点夹角计算函数
# -------------------------------
def angle_between_three_points(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# -------------------------------
# 从文件名提取信息
# -------------------------------
def parse_filename(filename):
    name_match = re.search(r'运动者\d+', filename)
    name = name_match.group() if name_match else "未知"
    return name


# -------------------------------
# 计算文件角度并整理成字典
# -------------------------------
def compute_file_metrics(file_path):
    df = pd.read_excel(file_path)

    # 去掉列名中的空格
    df.columns = df.columns.str.strip()

    print(f"\n📂 正在处理文件: {file_path}")
    print("列名检查：", df.columns.tolist())  # 打印列名，方便排错

    required_cols = ['23_X', '23_Y', '24_X', '24_Y',
                     '25_X', '25_Y', '26_X', '26_Y',
                     '27_X', '27_Y', '28_X', '28_Y',
                     '11_X', '11_Y', '12_X', '12_Y',
                     '19_X', '19_Y', '20_X', '20_Y',
                     '31_X', '31_Y', '32_X', '32_Y']

    # 检查缺失列
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ 文件 {file_path} 缺少列: {missing}，跳过计算")
        return None

    results = {}
    row1 = df.iloc[0]  # 取第一行作为示例

    # 发力阶段角度
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

    # 起跳阶段角度（这里为了演示，还是用 row1，你可以换成起跳帧）
    results['起跳膝角'] = round(hka, 2)
    results['起跳髋角'] = round(shk, 2)
    results['起跳踝角'] = round(jhx, 2)
    results['起跳肩角'] = round(wrist, 2)

    # 文件信息
    results['名字'] = parse_filename(file_path)

    return results


# -------------------------------
# 遍历所有文件
# -------------------------------
all_files = read_excel_files(pre_folder) + read_excel_files(post_folder)
all_data = [compute_file_metrics(f) for f in all_files]
all_data = [d for d in all_data if d is not None]  # 过滤掉缺失列的文件

# -------------------------------
# 保存到 Excel
# -------------------------------
df_out = pd.DataFrame(all_data)
cols_order = ['名字', '发力膝角', '发力髋角', '发力踝角', '发力肩角',
              '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角']
df_out = df_out[cols_order]
df_out.to_excel(output_file, index=False)

print(f"\n✅ 所有文件数据已保存到: {output_file}")
