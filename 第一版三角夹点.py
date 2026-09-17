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
    return [(f, pd.read_excel(f)) for f in all_files]  # 返回文件名和数据

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
# 倾斜角度计算函数（肩-髋连线 vs 竖直方向）
# -------------------------------
def tilt_angle(A, B):
    vec = np.array(A) - np.array(B)
    vertical = np.array([0, -1])  # 竖直向上
    cosine_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# -------------------------------
# 每个文件提取两类指标
# -------------------------------
def compute_file_metrics(filename, df):
    results = {}
    # ============= 第一行：关节夹角 =============
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
    wrist = angle_between_three_points(
        (st.mean([row1['19_X'], row1['20_X']]), st.mean([row1['19_Y'], row1['20_Y']])),
        (st.mean([row1['11_X'], row1['12_X']]), st.mean([row1['11_Y'], row1['12_Y']])),
        (st.mean([row1['23_X'], row1['24_X']]), st.mean([row1['23_Y'], row1['24_Y']]))
    )
    results.update({'髋膝踝': hka, '肩髋膝': shk, '腕摆动': wrist})

    # ============= 第二行：躯干倾斜（如果有） =============
    if len(df) > 1:
        row2 = df.iloc[1]
        shoulder = (st.mean([row2['11_X'], row2['12_X']]), st.mean([row2['11_Y'], row2['12_Y']]))
        hip = (st.mean([row2['23_X'], row2['24_X']]), st.mean([row2['23_Y'], row2['24_Y']]))
        tilt = tilt_angle(shoulder, hip)
        results['躯干倾斜'] = tilt
    else:
        results['躯干倾斜'] = np.nan

    print(f"\n文件: {os.path.basename(filename)}")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")
    return results

# -------------------------------
# 遍历所有文件并打印详细值
# -------------------------------
pre_metrics = pd.DataFrame([compute_file_metrics(f, df) for f, df in pre_files])
post_metrics = pd.DataFrame([compute_file_metrics(f, df) for f, df in post_files])

# -------------------------------
# T检验
# -------------------------------
results = {}
for col in pre_metrics.columns:
    t_stat, p_val = ttest_ind(pre_metrics[col].dropna(), post_metrics[col].dropna())
    results[col] = {'t统计量': t_stat, 'p值': p_val}

results_df = pd.DataFrame(results).T
print("\n================ T检验结果 ================ ")
print(results_df)

# -------------------------------
# 输出 p 值均值
# -------------------------------
mean_p = results_df['p值'].mean()
print("\n四类指标 p 值均值:", mean_p)
