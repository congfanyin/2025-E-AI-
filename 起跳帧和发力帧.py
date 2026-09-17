import os
import glob
import pandas as pd

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = r"D:\math\附件\附件3\姿势调整后"
output_folder = r"附件/附件3/起跳帧数据"
fps = 30  # 每秒帧数
cumulative_threshold = 40  # 足部累计下降阈值

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 读取文件列表
# -------------------------------
excel_files = glob.glob(os.path.join(pre_folder, "*.xlsx"))

# -------------------------------
# 处理每个文件
# -------------------------------
for file_path in excel_files:
    df = pd.read_excel(file_path)
    df_after_100 = df[df['帧号'] >= 100].reset_index(drop=True)

    jump_frame = None
    count = 0
    start_idx = None
    cumulative_diff = 0

    diffs = df_after_100[['31_Y', '32_Y']].diff()
    mask_down = (diffs < 0).all(axis=1)

    for i, v in enumerate(mask_down):
        if v:
            count += 1
            if start_idx is None:
                start_idx = i - 1
                cumulative_diff = 0
            frame_diff = (df_after_100.loc[i - 1, ['31_Y', '32_Y']] -
                          df_after_100.loc[i, ['31_Y', '32_Y']]).abs().sum()
            cumulative_diff += frame_diff
            if cumulative_diff >= cumulative_threshold:
                jump_frame = df_after_100.loc[i - 1, '帧号']
                break  # 找到第一个起跳帧立即停止
        else:
            count = 0
            start_idx = None
            cumulative_diff = 0

    if jump_frame is None:
        print(f"{os.path.basename(file_path)} 未检测到起跳帧，跳过保存")
        continue  # 未检测到起跳帧，不保存文件

    print(f"{os.path.basename(file_path)} 检测到第一个起跳帧：{jump_frame}")

    # 获取第一个起跳帧及前三秒数据
    frames_before_jump = fps * 0.33
    idx = df_after_100.index[df_after_100['帧号'] == jump_frame][0]
    start_idx = max(0, idx - frames_before_jump)
    jump_data = df_after_100.loc[start_idx:idx]

    # 保存到新的 Excel（仅当检测到起跳帧）
    output_path = os.path.join(output_folder, os.path.basename(file_path).replace(".xlsx", "_起跳帧.xlsx"))
    jump_data.to_excel(output_path, index=False)
    print(f"{os.path.basename(file_path)} 起跳帧数据已保存至：{output_path}\n")
