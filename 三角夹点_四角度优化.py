import pandas as pd
import numpy as np
import statistics as st

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
# 从行数据计算角度
# -------------------------------
def compute_angles_from_row(row, prefix=""):
    hka = angle_between_three_points(
        (st.mean([row['24_X'], row['23_X']]), st.mean([row['24_Y'], row['23_Y']])),
        (st.mean([row['26_X'], row['25_X']]), st.mean([row['26_Y'], row['25_Y']])),
        (st.mean([row['28_X'], row['27_X']]), st.mean([row['28_Y'], row['27_Y']]))
    )
    shk = angle_between_three_points(
        (st.mean([row['11_X'], row['12_X']]), st.mean([row['11_Y'], row['12_Y']])),
        (st.mean([row['23_X'], row['24_X']]), st.mean([row['23_Y'], row['24_Y']])),
        (st.mean([row['25_X'], row['26_X']]), st.mean([row['25_Y'], row['26_Y']]))
    )
    jhx = angle_between_three_points(
        (st.mean([row['25_X'], row['26_X']]), st.mean([row['25_Y'], row['26_Y']])),
        (st.mean([row['27_X'], row['28_X']]), st.mean([row['27_Y'], row['28_Y']])),
        (st.mean([row['31_X'], row['32_X']]), st.mean([row['31_Y'], row['32_Y']]))
    )
    wrist = angle_between_three_points(
        (st.mean([row['19_X'], row['20_X']]), st.mean([row['19_Y'], row['20_Y']])),
        (st.mean([row['11_X'], row['12_X']]), st.mean([row['11_Y'], row['12_Y']])),
        (st.mean([row['23_X'], row['24_X']]), st.mean([row['23_Y'], row['24_Y']]))
    )

    return {
        f"{prefix}膝角": round(hka, 2),
        f"{prefix}髋角": round(shk, 2),
        f"{prefix}踝角": round(jhx, 2),
        f"{prefix}肩角": round(wrist, 2)
    }

# -------------------------------
# 单文件分析函数
# -------------------------------
def analyze_single_file(file_path):
    df = pd.read_excel(file_path)
    results = {}

    # 下蹲角度（第1帧）
    row1 = df.iloc[0]
    results.update(compute_angles_from_row(row1, prefix="发力"))

    # 起跳角度（第2帧，如果没有则用第1帧）
    row2 = df.iloc[1] if len(df) > 1 else df.iloc[0]
    results.update(compute_angles_from_row(row2, prefix="起跳"))

    # 躯干倾斜（使用第2帧，如果没有则设NaN）
    if len(df) > 1:
        shoulder = (st.mean([row2['11_X'], row2['12_X']]), st.mean([row2['11_Y'], row2['12_Y']]))
        hip = (st.mean([row2['23_X'], row2['24_X']]), st.mean([row2['23_Y'], row2['24_Y']]))
        results['躯干倾斜'] = round(tilt_angle(shoulder, hip), 2)
    else:
        results['躯干倾斜'] = np.nan

    # 打印结果
    for k, v in results.items():
        print(f"{k}: {v}")

# -------------------------------
# 示例调用
# -------------------------------
file_path = "附件/附件5/11.xlsx"
analyze_single_file(file_path)
