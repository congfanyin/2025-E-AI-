import pandas as pd

# 1. 读取数据
source_file = "附件/附件4.xlsx"  # 含完整数据
target_file = "副本跳远角度汇总.xlsx"  # 目标文件
output_file = "跳远角度填充完成.xlsx"  # 输出文件

df_source = pd.read_excel(source_file)
df_target = pd.read_excel(target_file)

# 2. 确定匹配列和填充列
columns_to_fill = [
    "脂肪控制量 (kg)",
    "身高 (cm)",
    "肌肉重量 (kg)",
    "骨骼肌重量 (kg)",
    "标准体重 (kg)"
]

# 3. 按姓名匹配填充数据
for col in columns_to_fill:
    # 检查目标表是否有该列，没有则创建空列
    if col not in df_target.columns:
        df_target[col] = pd.NA

    # 用源表的值填充目标表
    df_target[col] = df_target["姓名"].map(
        df_source.set_index("姓名")[col]
    )

# 4. 保存结果
df_target.to_excel(output_file, index=False)
print(f"✅ 已完成填充，结果保存为 {output_file}")
