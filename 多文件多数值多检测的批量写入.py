import pandas as pd
import numpy as np
import statistics as st
import os
import re

# ------------------------------- 文件路径设置 -------------------------------
input_file = r"附件/附件5/运动者11的跳远位置信息.xlsx"
output_file = r"单文件角度结果.xlsx"
muscle_file = "附件4.xlsx"

# ------------------------------- 工具函数 -------------------------------
def angle_between_three_points(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def tilt_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(np.degrees(np.arctan2(dy, dx)))  # 绝对值，忽略方向

def parse_filename(file_path):
    """
    返回 文件名的前三个字符 + 紧接着的数字
    例如：'运动者11跳远.xlsx' -> '运动者11'
    """
    fname = os.path.basename(file_path)
    fname_no_ext = os.path.splitext(fname)[0]
    match = re.match(r'^(.{3}\d+)', fname_no_ext)
    if match:
        return match.group(1)
    else:
        return fname_no_ext

# ------------------------------- 跳远起跳帧检测 -------------------------------
def detect_jump_frames(df):
    df_after_100 = df[df['帧号'] >= 100].reset_index(drop=True)
    jump_frame = None
    count = 0
    start_idx = None

    diffs = df_after_100[['29_Y', '30_Y', '31_Y', '32_Y']].diff()
    mask_down = (diffs < 0).all(axis=1)

    for i, v in enumerate(mask_down):
        if v:
            count += 1
            if start_idx is None:
                start_idx = i - 1
        else:
            if count >= 2 and start_idx is not None:
                total_diff = (df_after_100.loc[i-1, ['29_Y','30_Y','31_Y','32_Y']] -
                              df_after_100.loc[start_idx, ['29_Y','30_Y','31_Y','32_Y']]).abs().sum()
                if total_diff > 50:
                    jump_frame = df_after_100.loc[start_idx, '帧号']
                    break
            count = 0
            start_idx = None

    pre10_frame = None
    if jump_frame is not None:
        jump_idx = df.index[df['帧号'] == jump_frame][0]
        pre10_idx = max(jump_idx - 10, 0)
        pre10_frame = df.loc[pre10_idx, '帧号']

    frames_to_use = [pre10_frame, jump_frame]
    df_selected = df[df['帧号'].isin(frames_to_use)].copy()
    return df_selected, pre10_frame, jump_frame

# ------------------------------- 计算角度指标 -------------------------------
def compute_file_metrics(file_path):
    df = pd.read_excel(file_path)
    df_frames, pre10_frame, jump_frame = detect_jump_frames(df)

    # 打印帧号
    print(f"第一个帧（计算角度）: {pre10_frame}, 第二个帧（计算躯干倾斜）: {jump_frame}")

    results = {}

    # 第一个帧：关节角度
    row_angle = df_frames.iloc[0]
    results['髋膝踝'] = round(angle_between_three_points(
        (st.mean([row_angle['24_X'], row_angle['23_X']]), st.mean([row_angle['24_Y'], row_angle['23_Y']])),
        (st.mean([row_angle['26_X'], row_angle['25_X']]), st.mean([row_angle['26_Y'], row_angle['25_Y']])),
        (st.mean([row_angle['28_X'], row_angle['27_X']]), st.mean([row_angle['28_Y'], row_angle['27_Y']])),
    ), 2)
    results['肩髋膝'] = round(angle_between_three_points(
        (st.mean([row_angle['11_X'], row_angle['12_X']]), st.mean([row_angle['11_Y'], row_angle['12_Y']])),
        (st.mean([row_angle['23_X'], row_angle['24_X']]), st.mean([row_angle['23_Y'], row_angle['24_Y']])),
        (st.mean([row_angle['25_X'], row_angle['26_X']]), st.mean([row_angle['25_Y'], row_angle['26_Y']])),
    ), 2)
    results['脚踝膝'] = round(angle_between_three_points(
        (st.mean([row_angle['25_X'], row_angle['26_X']]), st.mean([row_angle['25_Y'], row_angle['26_Y']])),
        (st.mean([row_angle['27_X'], row_angle['28_X']]), st.mean([row_angle['27_Y'], row_angle['28_Y']])),
        (st.mean([row_angle['31_X'], row_angle['32_X']]), st.mean([row_angle['31_Y'], row_angle['32_Y']])),
    ), 2)
    results['腕摆动'] = round(angle_between_three_points(
        (st.mean([row_angle['19_X'], row_angle['20_X']]), st.mean([row_angle['19_Y'], row_angle['20_Y']])),
        (st.mean([row_angle['11_X'], row_angle['12_X']]), st.mean([row_angle['11_Y'], row_angle['12_Y']])),
        (st.mean([row_angle['23_X'], row_angle['24_X']]), st.mean([row_angle['23_Y'], row_angle['24_Y']])),
    ), 2)

    # 第二个帧：躯干倾斜（绝对值）
    if len(df_frames) > 1:
        row_tilt = df_frames.iloc[1]
        shoulder = (st.mean([row_tilt['11_X'], row_tilt['12_X']]), st.mean([row_tilt['11_Y'], row_tilt['12_Y']]))
        hip = (st.mean([row_tilt['23_X'], row_tilt['24_X']]), st.mean([row_tilt['23_Y'], row_tilt['24_Y']]))
        results['躯干倾斜'] = round(tilt_angle(shoulder, hip), 2)
    else:
        results['躯干倾斜'] = np.nan

    # 名字：前三个字符+紧接数字
    results['姓名'] = parse_filename(file_path)

    return results

# ------------------------------- 读取肌肉数据并写入 -------------------------------
def add_muscle_data(df_result, muscle_file):
    muscle_df = pd.read_excel(muscle_file)
    columns_to_add = ['骨骼肌重量 (kg)', '基础代谢 (kcal)', '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量 (kg)']

    for idx, row in df_result.iterrows():
        name = row['姓名']
        match = muscle_df[muscle_df['姓名'] == name]
        if not match.empty:
            for col in columns_to_add:
                df_result.at[idx, col] = match.iloc[0][col]
        else:
            for col in columns_to_add:
                df_result.at[idx, col] = np.nan
    return df_result

# ------------------------------- 单文件处理 -------------------------------
result = compute_file_metrics(input_file)
df_out = pd.DataFrame([result])
cols_order = ['姓名', '髋膝踝', '肩髋膝', '脚踝膝', '腕摆动', '躯干倾斜']
df_out = df_out[cols_order]

# 加入肌肉数据
df_out = add_muscle_data(df_out, muscle_file)

# 打印表格
print("\n最终结果表格：")
print(df_out.to_string(index=False))

# 保存到Excel
df_out.to_excel(output_file, index=False)
