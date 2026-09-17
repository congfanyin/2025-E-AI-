import os
import glob
import pandas as pd
import re
import traceback

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = r"附件/附件3/调整前(平滑)"   # 输入文件夹
output_path = os.path.join(pre_folder, "调整前_平滑_优化.xlsx")  # 输出 Excel
fps = 30  # 每秒帧数
threshold_jump = 5  # 起跳帧判断阈值
threshold_land = 5  # 落地帧判断阈值
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
# 起跳帧 & 落地帧检测函数（平滑 + 80%窗口判定）
# -------------------------------
def get_jump_and_land(df, fps=50, threshold_jump=5, threshold_land=5,
                      consecutive=5, threshold_ratio=0.8, search_start=0, search_end=None):
    jump_frame, land_frame = None, None
    if df.empty or len(df) < 50:
        return jump_frame, land_frame

    # 平滑处理
    smooth_vals = smooth_df(df, required_cols, window=5)
    baseline = smooth_vals.iloc[:50].mean()

    if search_end is None:
        search_end = len(df)

    df_search = smooth_vals.iloc[search_start:search_end].reset_index(drop=True)
    frame_search = df['帧号'].iloc[search_start:search_end].reset_index(drop=True)

    # -------- 起跳帧判定 --------
    for i in range(len(df_search) - consecutive + 1):
        window = df_search.iloc[i:i+consecutive]
        exceed_count = (window - baseline).abs() > threshold_jump
        if (exceed_count.sum(axis=1) / len(required_cols) >= threshold_ratio).all():
            jump_frame = int(frame_search.iloc[i])
            break

    # -------- 落地帧判定 --------
    if jump_frame is not None:
        df_after_jump = smooth_vals[df['帧号'] > jump_frame].reset_index(drop=True)
        frame_after_jump = df['帧号'][df['帧号'] > jump_frame].reset_index(drop=True)

        for i in range(len(df_after_jump)):
            current_mean = df_after_jump.iloc[i].mean()
            if abs(current_mean - baseline.mean()) <= threshold_land:
                land_frame = int(frame_after_jump.iloc[i])
                break

    return jump_frame, land_frame

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
                "落地帧": None,
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
                "落地帧": None,
                "错误": "数据不足50帧"
            })
            continue

        # -------- 计算起跳帧和落地帧 --------
        jump, land = get_jump_and_land(
            df,
            fps=fps,
            threshold_jump=threshold_jump,
            threshold_land=threshold_land,
            consecutive=consecutive_frames,
            threshold_ratio=threshold_ratio,
            search_start=search_start,
            search_end=search_end
        )

        results.append({
            "文件名": os.path.basename(file_path),
            "起跳帧": jump,
            "落地帧": land
        })
        print(f"{os.path.basename(file_path)} -> 起跳帧: {jump}, 落地帧: {land}")

    except Exception as e:
        print(f"{os.path.basename(file_path)} -> 处理出错:\n{traceback.format_exc()}")
        results.append({
            "文件名": os.path.basename(file_path),
            "起跳帧": None,
            "落地帧": None,
            "错误": str(e)
        })

# 保存结果
df_results = pd.DataFrame(results)[["文件名", "起跳帧", "落地帧"]]
df_results.to_excel(output_path, index=False)

print(f"\n✅ 批量处理完成，结果已保存 {output_path}")
