import os
import glob
import pandas as pd
import re
import traceback

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = r"D:\math\附件\附件1"   # 输入文件夹
output_path = os.path.join(pre_folder, "调整前_平滑_优化.xlsx")  # 输出 Excel
fps = 30  # 每秒帧数
threshold_jump = 5  # 起跳帧判断阈值
consecutive_frames = 5  # 连续帧数
threshold_ratio = 0.8   # 窗口内满足条件比例
search_start = 50       # 起跳帧搜索起始帧
search_end = None       # 起跳帧搜索结束帧（None表示搜索到最后一帧）

required_cols = ['29_Y','30_Y','31_Y','32_Y']  # 必要列

# -------------------------------
# 列名清洗函数
# -------------------------------
def clean_columns(columns):
    new_cols = []
    for col in columns:
        col_clean = str(col).strip()               # 去掉首尾空格
        col_clean = col_clean.replace('\xa0', '') # 去掉不可见空格
        col_clean = re.sub(r'\s+', '', col_clean) # 去掉中间空格
        new_cols.append(col_clean)
    return new_cols

# -------------------------------
# 数据平滑函数
# -------------------------------
def smooth_df(df, cols, window=5):
    return df[cols].rolling(window=window, min_periods=1, center=True).mean()

# -------------------------------
# 起跳帧 & 发力帧检测函数
# -------------------------------
def get_jump_and_force(df, fps=50, threshold_jump=5,
                       consecutive=5, threshold_ratio=0.8, search_start=0, search_end=None):
    jump_frame, force_frame = None, None
    if df.empty or len(df) < 50:
        return jump_frame, force_frame

    # 平滑处理
    smooth_vals = smooth_df(df, required_cols, window=5)
    baseline = smooth_vals.iloc[:50].mean()

    if search_end is None:
        search_end = len(df)

    df_search = smooth_vals.iloc[search_start:search_end].reset_index(drop=True)
    frame_search = df['帧号'].iloc[search_start:search_end].reset_index(drop=True)

    # -------- 起跳帧判定 --------
    for i in range(len(df_search) - consecutive + 1):
        window_vals = df_search.iloc[i:i+consecutive]
        exceed_count = (window_vals - baseline).abs() > threshold_jump
        if (exceed_count.sum(axis=1) / len(required_cols) >= threshold_ratio).all():
            jump_frame = int(frame_search.iloc[i])
            break

    # -------- 发力帧判定 --------
    if jump_frame is not None:
        wrist_cols = ['15_X', '16_X']
        if not all(col in df.columns for col in wrist_cols):
            print("缺少手腕点列，无法计算发力帧")
            force_frame = None
        else:
            # 起跳帧之前的手腕点
            wrist_vals = df.loc[df['帧号'] < jump_frame, wrist_cols]
            wrist_mean = wrist_vals.mean().mean()  # 所有手腕点均值

            # 从起跳帧向前找小于均值的第3帧
            wrist_before_jump = df.loc[df['帧号'] <= jump_frame, wrist_cols].reset_index(drop=True)
            frames_before_jump = df.loc[df['帧号'] <= jump_frame, '帧号'].reset_index(drop=True)
            count = 0
            for i in range(len(wrist_before_jump)-1, -1, -1):
                if wrist_before_jump.iloc[i].mean() < wrist_mean:  # 修改条件：小于均值
                    count += 1
                    if count == 3:  # 第三帧
                        force_frame = int(frames_before_jump.iloc[i])
                        break

    return jump_frame, force_frame

# -------------------------------
# 主程序：批量处理文件
# -------------------------------
results = []

for file_path in glob.glob(os.path.join(pre_folder, "*.xlsx")):
    try:
        df = pd.read_excel(file_path)

        # -------- 清洗列名 --------
        df.columns = clean_columns(df.columns)

        # -------- 检查必要列 --------
        if not all(col in df.columns for col in required_cols + ['帧号']):
            print(f"{os.path.basename(file_path)} -> 缺少必要列或帧号列, 跳过")
            results.append({
                "文件名": os.path.basename(file_path),
                "起跳帧": None,
                "发力帧": None,
                "错误": "缺少必要列或帧号列"
            })
            continue

        # -------- 强制数值化 --------
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
        df['帧号'] = pd.to_numeric(df['帧号'], errors='coerce')

        # -------- 丢弃包含 NaN 的行 --------
        df.dropna(subset=required_cols + ['帧号'], inplace=True)
        if len(df) < 50:
            print(f"{os.path.basename(file_path)} -> 数据不足50帧, 跳过")
            results.append({
                "文件名": os.path.basename(file_path),
                "起跳帧": None,
                "发力帧": None,
                "错误": "数据不足50帧"
            })
            continue

        # -------- 计算起跳帧和发力帧 --------
        jump, force = get_jump_and_force(
            df,
            fps=fps,
            threshold_jump=threshold_jump,
            consecutive=consecutive_frames,
            threshold_ratio=threshold_ratio,
            search_start=search_start,
            search_end=search_end
        )

        results.append({
            "文件名": os.path.basename(file_path),
            "起跳帧": jump,
            "发力帧": force
        })
        print(f"{os.path.basename(file_path)} -> 起跳帧: {jump}, 发力帧: {force}")

    except Exception as e:
        print(f"{os.path.basename(file_path)} -> 处理出错:\n{traceback.format_exc()}")
        results.append({
            "文件名": os.path.basename(file_path),
            "起跳帧": None,
            "发力帧": None,
            "错误": str(e)
        })

# 保存结果
df_results = pd.DataFrame(results)[["文件名", "起跳帧", "发力帧"]]
df_results.to_excel(output_path, index=False)

print(f"\n✅ 批量处理完成，结果已保存 {output_path}")
