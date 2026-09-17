import os
import glob
import pandas as pd
from scipy.signal import savgol_filter

# -------------------------------
# 参数设置
# -------------------------------
pre_folder = r"附件/附件5"
output_folder = r"D:\math\附件\附件3\调整后(平滑)"
os.makedirs(output_folder, exist_ok=True)

smooth_rounds = 3  # 平滑迭代次数，可调整：2~5 较合适

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

    # 逐列平滑
    polyorder = 2
    for col in cols:
        if col in df.columns:
            series = df[col].ffill().bfill().values
            n = len(series)

            if n < 3:  # 数据太短无法平滑
                continue

            # 计算合适的窗口长度：不超过数据长度，且为奇数
            window_length = min(5, n if n % 2 == 1 else n - 1)
            if window_length < polyorder + 2:
                window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
                if window_length > n:  # 如果还是大于数据长度，就跳过
                    continue

            # 反复平滑
            for _ in range(smooth_rounds):
                series = savgol_filter(series, window_length=window_length, polyorder=polyorder)
            df[col] = series

    # 保存修正结果
    out_path = os.path.join(output_folder, os.path.basename(file_path).replace(".xlsx", "_修正.xlsx"))
    df.to_excel(out_path, index=False)
    print(f"{os.path.basename(file_path)} 处理完成，保存到：{out_path}")
