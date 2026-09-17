import pandas as pd

# 读取原始 Excel
df = pd.read_excel("副本跳远角度汇总1（改）.xlsx")

# 将数据复制多次，比如复制5倍
n_copies = 500
df_expanded = pd.concat([df]*n_copies, ignore_index=True)

# 保存到新文件
df_expanded.to_excel("副本跳远角度汇总1（改）_扩展.xlsx", index=False)
print(f"✅ 数据已扩展 {n_copies} 倍，保存为 新文件")
