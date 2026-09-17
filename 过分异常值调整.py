import pandas as pd

# 读取 Excel
df = pd.read_excel(r"D:\math\附件\附件1\运动者2的跳远位置信息.xlsx")

df_corrected = df.copy()

# 遍历每一帧
for i in range(len(df)):
    # ---------------- 脚跟 ----------------
    heel_vals = df.loc[i, ['29_Y', '30_Y']].values
    heel_diff = abs(heel_vals[0] - heel_vals[1])
    if heel_diff > 30:
        # 找到较大和较小的值
        big_idx = '29_Y' if df.loc[i, '29_Y'] > df.loc[i, '30_Y'] else '30_Y'
        small_idx = '30_Y' if big_idx == '29_Y' else '29_Y'
        # 修正：靠近另一只脚 ±10
        df_corrected.loc[i, big_idx] = df.loc[i, small_idx] + 10 * (1 if df.loc[i, big_idx] > df.loc[i, small_idx] else -1)

    # ---------------- 脚尖 ----------------
    toe_vals = df.loc[i, ['31_Y', '32_Y']].values
    toe_diff = abs(toe_vals[0] - toe_vals[1])
    if toe_diff > 30:
        big_idx = '31_Y' if df.loc[i, '31_Y'] > df.loc[i, '32_Y'] else '32_Y'
        small_idx = '32_Y' if big_idx == '31_Y' else '31_Y'
        df_corrected.loc[i, big_idx] = df.loc[i, small_idx] + 10 * (1 if df.loc[i, big_idx] > df.loc[i, small_idx] else -1)

# 保存修正后的数据
df_corrected.to_excel(r"D:\math\附件\附件1\运动者2 _corrected.xlsx", index=False)

print("修正完成，已保存运动员2的修正数据。")
