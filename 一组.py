import pandas as pd

# 读取 Excel
df = pd.read_excel(r"D:\math\附件\附件1\运动者1_corrected.xlsx")

# ---------------- 起跳帧 ----------------
df_after_100 = df[df['帧号'] >= 100].reset_index(drop=True)

jump_frame = None
count = 0
start_idx = None
cumulative_diff = 0

diffs = df_after_100[['29_Y', '30_Y', '31_Y', '32_Y']].diff()
mask_down = (diffs < 0).all(axis=1)

for i, v in enumerate(mask_down):
    if v:
        count += 1
        if start_idx is None:
            start_idx = i - 1
            cumulative_diff = 0
        frame_diff = (df_after_100.loc[i - 1, ['29_Y', '30_Y', '31_Y', '32_Y']] -
                      df_after_100.loc[i, ['29_Y', '30_Y', '31_Y', '32_Y']]).abs().sum()
        cumulative_diff += frame_diff
        if cumulative_diff >= 100:
            jump_frame = df_after_100.loc[i - 1, '帧号']
            break
    else:
        count = 0
        start_idx = None
        cumulative_diff = 0

# ---------------- 顶峰帧 ----------------
df_after_jump = df[df['帧号'] >= jump_frame].reset_index(drop=True)

d1 = df_after_jump['31_Y'] - df_after_jump['29_Y']
d2 = df_after_jump['32_Y'] - df_after_jump['30_Y']
ref = pd.concat([d1, d2], axis=1).max(axis=1)  # 最大差值作为参考

apex_frame = None
for i in range(2, len(ref) - 2):
    if ref[i] < ref[i - 2] and ref[i] < ref[i - 1] and ref[i] < ref[i + 1] and ref[i] < ref[i + 2]:
        apex_frame = df_after_jump.loc[i, '帧号']
        break

# ---------------- 落地帧 ----------------
# 起跳前均值（每列单独平均）
pre_jump_df = df[df['帧号'] < jump_frame]
pre_jump_mean = pre_jump_df[['29_Y', '30_Y', '31_Y', '32_Y']].mean()  # 每列平均
print("起跳前四列平均值:\n", pre_jump_mean)

# 落地帧搜索范围
search_range = df[(df['帧号'] > jump_frame) & (df['帧号'] <= 200)].copy()

# 初筛：差值条件
d1_abs = abs(search_range['31_Y'] - search_range['29_Y'])
d2_abs = abs(search_range['32_Y'] - search_range['30_Y'])
mask_land = (d1_abs < 3) & (d2_abs < 3)

if mask_land.any():
    # 只取第一个满足差值条件的帧
    first_idx = mask_land.idxmax()  # 第一个True对应的原始索引
    candidate_frame = search_range.loc[first_idx]

    land_frame = candidate_frame['帧号']
    land_values = candidate_frame[['29_Y', '30_Y', '31_Y', '32_Y']].values
    per_column_diff = candidate_frame[['29_Y', '30_Y', '31_Y', '32_Y']].sub(pre_jump_mean).abs()
else:
    land_frame = None
    land_values = None
    per_column_diff = None

print("起跳帧:", jump_frame)
print("顶峰帧:", apex_frame)
print("落地帧:", land_frame)
print("落地帧对应数值:", land_values)
print("落地帧与起跳前每列均值差值:\n", per_column_diff)
