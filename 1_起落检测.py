import pandas as pd

# 读取 Excel
df = pd.read_excel(r"附件/附件1/运动者1的跳远位置信息_修正.xlsx")
# df = pd.read_excel(r"附件/附件1/运动者2_修正后.xlsx")

# ---------------- 起跳帧 ----------------
df_after_100 = df[df['帧号'] >= 100].reset_index(drop=True)

jump_frame = None
count = 0
start_idx = None
cumulative_diff = 0

diffs = df_after_100[['29_Y', '30_Y', '31_Y', '32_Y']].diff()
mask_down = (diffs < 0).all(axis=1)

for i, v in enumerate(mask_down):
    if v:  # 当前帧满足下降
        count += 1
        if start_idx is None:
            # 记录连续下降的起点
            start_idx = i - 1
            cumulative_diff = 0

        # 累积下降值
        frame_diff = (df_after_100.loc[i - 1, ['29_Y', '30_Y', '31_Y', '32_Y']] -
                      df_after_100.loc[i, ['29_Y', '30_Y', '31_Y', '32_Y']]).abs().sum()
        cumulative_diff += frame_diff


        if count >= 3 and cumulative_diff > 50:
            jump_frame = df_after_100.loc[start_idx, '帧号']
            break
    else:
        # 不再下降 → 重置
        count = 0
        start_idx = None
        cumulative_diff = 0


        # ---------------- 落地帧 ----------------
# 起跳前四列均值
pre_jump_df = df[df['帧号'] < jump_frame]
pre_jump_mean = pre_jump_df[['29_Y', '30_Y', '31_Y', '32_Y']].mean()
print("起跳前四列均值:\n", pre_jump_mean)

# 落地帧搜索范围
search_range = df[(df['帧号'] > jump_frame) & (df['帧号'] <= 190)].copy()

if jump_frame is not None:
    df_after_jump = search_range.reset_index(drop=True)

    # 条件1：计算差值
    d1_abs = abs(df_after_jump['31_Y'] - df_after_jump['29_Y'])
    d2_abs = abs(df_after_jump['32_Y'] - df_after_jump['30_Y'])
    mask_land = (d1_abs < 3) & (d2_abs < 3)

    if mask_land.any():
        # 满足条件1的帧
        candidate_frames = df_after_jump[mask_land].copy()

        # 条件2：选取最接近起跳前均值的帧
        diff_cols = candidate_frames[['29_Y', '30_Y', '31_Y', '32_Y']].sub(pre_jump_mean).abs()
        candidate_frames['mean_diff'] = diff_cols.sum(axis=1)

        land_frame_row = candidate_frames.loc[candidate_frames['mean_diff'].idxmin()]
        land_frame = land_frame_row['帧号']
        land_values = land_frame_row[['29_Y', '30_Y', '31_Y', '32_Y']].values
        per_column_diff = land_frame_row[['29_Y', '30_Y', '31_Y', '32_Y']].sub(pre_jump_mean).abs()
    else:
        land_frame = None
        land_values = None
        per_column_diff = None
else:
    land_frame = None
    land_values = None
    per_column_diff = None

print("起跳帧:", jump_frame)
print("落地帧:", land_frame)
print("落地帧对应数值:", land_values)
print("落地帧与起跳前每列均值差值:\n", per_column_diff)
