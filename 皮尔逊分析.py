import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 读取 Excel
df = pd.read_excel("跳远角度汇总1.xlsx")

# 设置目标列为第一列
target = "成绩"

# 自动取 target 后面的所有列作为 features
# features = "髋膝踝","肩髋膝","腕摆动","躯干倾斜","骨骼肌重量 (kg)","脂肪控制量 (kg)","身高 (cm)",'肌肉重量 (kg)','标准体重 (kg)'
features = '下蹲髋膝踝', '下蹲肩髋膝', '下蹲脚踝膝', '下蹲腕肩髋', '起跳髋膝踝', '起跳肩髋膝', '起跳脚踝膝', '起跳腕肩髋',"骨骼肌重量 (kg)", "基础代谢(kcal)", "肌肉率 (%)", '去脂体重 (kg)','肌肉重量(kg)'

# 转换为数值型，非数值变 NaN
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=[target] + list(features))

results = []

for col in features:
    x = df[target].astype(float)
    y = df[col].astype(float)

    # 保留两列都非空的行
    mask = x.notna() & y.notna()
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 2:
        print(f"列 {col} 数据不足，无法计算相关性")
        continue  # 跳过这个特征

    # Pearson
    r, p = pearsonr(x_valid, y_valid)
    # Spearman
    rho, p_s = spearmanr(x_valid, y_valid)

    results.append({
        "变量": col,
        "Pearson r": round(r, 6),
        "Pearson p值": round(p, 6),
        "Spearman ρ": round(rho, 6),
        "Spearman p值": round(p_s, 6)
    })
result_df = pd.DataFrame(results)
print(result_df)