import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import statistics as st

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = "附件/附件3/起跳帧数据/调整前"
post_folder = "附件/附件3/起跳帧数据/调整后"

# -------------------------------
# 读取 Excel 文件
# -------------------------------
def read_excel_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    return [pd.read_excel(f) for f in all_files]

pre_files = read_excel_files(pre_folder)
post_files = read_excel_files(post_folder)

# -------------------------------
# 三点夹角计算函数
# -------------------------------
def angle_between_three_points(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# -------------------------------
# 计算每帧角度
# -------------------------------
def compute_angles(df):
    angles = []
    for _, row in df.iterrows():
        try:
            # 髋膝踝左右取平均
            hka = angle_between_three_points(
                (st.mean([row['24_X'], row['23_X']]), st.mean([row['24_Y'], row['23_Y']])),
                (st.mean([row['26_X'], row['25_X']]), st.mean([row['26_Y'], row['25_Y']])),
                (st.mean([row['28_X'], row['27_X']]), st.mean([row['28_Y'], row['27_Y']]))
            )

            hip_knee_ankle = hka

            # 肩髋膝左右取平均
            shk = angle_between_three_points(
                (st.mean([row['11_X'], row['12_X']]), st.mean([row['11_Y'], row['12_Y']])),
                (st.mean([row['23_X'], row['24_X']]), st.mean([row['23_Y'], row['24_Y']])),
                (st.mean([row['25_X'], row['26_X']]), st.mean([row['25_Y'], row['26_Y']]))
            )

            shoulder_hip_knee = shk

            jhx = angle_between_three_points(
                (st.mean([row['25_X'], row['26_X']]), st.mean([row['25_Y'], row['26_Y']])),
                (st.mean([row['27_X'], row['28_X']]), st.mean([row['27_Y'], row['28_Y']])),
                (st.mean([row['31_X'], row['32_X']]), st.mean([row['31_Y'], row['32_Y']]))

            )

            jhx_hip_knee = jhx

            # 腕摆动左右取平均
            wrist = angle_between_three_points(
                (st.mean([row['19_X'], row['20_X']]), st.mean([row['19_Y'], row['20_Y']])),
                (st.mean([row['11_X'], row['12_X']]), st.mean([row['11_Y'], row['12_Y']])),
                (st.mean([row['23_X'], row['24_X']]), st.mean([row['23_Y'], row['24_Y']]))
            )

            wrist_swing = wrist

            angles.append([hip_knee_ankle, shoulder_hip_knee,jhx_hip_knee, wrist_swing])
        except KeyError as e:
            print(f"缺少关键点列: {e}")
            continue

    return pd.DataFrame(angles, columns=['髋膝踝', '肩髋膝', '脚踝膝', '腕摆动'])

# -------------------------------
# 合并所有文件数据
# -------------------------------
pre_angles = pd.concat([compute_angles(df) for df in pre_files], ignore_index=True)
post_angles = pd.concat([compute_angles(df) for df in post_files], ignore_index=True)

# -------------------------------
# T检验分析
# -------------------------------
results = {}
for col in pre_angles.columns:
    t_stat, p_val = ttest_ind(pre_angles[col], post_angles[col])
    print(pre_angles)
    results[col] = {'t统计量': t_stat, 'p值': p_val}

results_df = pd.DataFrame(results).T
print("T检验结果（姿势调整前 vs 姿势调整后）:")
print(results_df)

# -------------------------------
# 输出 p 值均值
# -------------------------------
mean_p = results_df['p值'].mean()
print("\n三类指标 p 值均值:", mean_p)
