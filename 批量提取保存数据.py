import pandas as pd
import numpy as np

def excel_rows_to_txt_with_comma(file_path, start_col='C', end_col='P', start_row=2, end_row=1000, txt_path='output.txt'):
    """
    读取 Excel 指定区域，每行数据保存为一个列表，并在每行结尾加逗号，写入 UTF-8 编码 txt 文件。

    参数:
    - file_path: Excel 文件路径
    - start_col, end_col: 列范围，字母表示（如 'C', 'P'）
    - start_row, end_row: 行范围，Excel 从1开始计数
    - txt_path: 输出 txt 文件路径
    """
    # 读取 Excel，不使用表头
    df = pd.read_excel(file_path, header=None)

    # 列字母转换为索引
    start_idx = ord(start_col.upper()) - ord('A')
    end_idx = ord(end_col.upper()) - ord('A') + 1  # 右开区间

    # 行索引转换为 Python 索引
    start_row_idx = start_row - 1
    end_row_idx = end_row

    # 获取指定区域的数据
    data_slice = df.iloc[start_row_idx:end_row_idx, start_idx:end_idx]

    # 去掉空值并转换为 float64，每行作为一个列表
    rows_list = []
    for _, row in data_slice.iterrows():
        row_list = row.dropna().astype(np.float64).tolist()
        rows_list.append(row_list)

    # 保存到 txt 文件，每行一个列表，结尾加逗号
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row_list in rows_list:
            f.write(str(row_list) + ',\n')

    print(f"数据已保存到 {txt_path}")
    return rows_list

# 示例调用
rows = excel_rows_to_txt_with_comma('运动者11_体质信息(2)_预测完成.xlsx', start_col='B', end_col='P', start_row=2, end_row=10001, txt_path='output.txt')
for r in rows[:5]:  # 打印前5行示例
    print(r)
