import os
import glob
import pandas as pd
from scipy.signal import savgol_filter

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = r"D:\math\附件\附件3\姿势调整后"
output_folder = r"D:\math\附件\附件3\调整后(平滑)"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 处理每个文件
# -------------------------------
excel_files = glob.glob(os.path.join(pre_folder, "*.xlsx"))

for file_path in excel_files:
    df = pd.read_excel(file_path)

    # 清理列名（去掉空格和奇怪字符）
    df.columns = df.columns.str.strip()

    # 获取需要处理的列（除“帧号”列）
    cols = [c for c in df.columns if c != '帧号']

    # 替换 0 → 上下行平均值
    for col in cols:
        if col in df.columns:
            mask = df[col] == 0
            df.loc[mask, col] = (df[col].shift(1) + df[col].shift(-1)) / 2

    # 对每列做 Savitzky–Golay 平滑
    # window_length 必须为奇数，polyorder 一般取 2 或 3
    window_length = 5 if len(df) >= 5 else (len(df) // 2 * 2 + 1)  # 自动调整
    polyorder = 2
    for col in cols:
        if col in df.columns:
            # 忽略空值或非数值
            df[col] = savgol_filter(df[col].ffill().bfill(),
                                    window_length=window_length, polyorder=polyorder)

    # 保存修正结果
    out_path = os.path.join(output_folder, os.path.basename(file_path).replace(".xlsx", "_修正.xlsx"))
    df.to_excel(out_path, index=False)
    print(f"{os.path.basename(file_path)} 处理完成，保存到：{out_path}")
