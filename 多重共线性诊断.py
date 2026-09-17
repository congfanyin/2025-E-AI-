import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimHei")
# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("跳远角度汇总1.xlsx")
df.columns = df.columns.str.strip()  # 去掉列名空格

target = "成绩"

# =========================
# 2. 提取数值型特征与目标
# =========================
X = df.drop(columns=[target])
X = X.select_dtypes(include=[np.number])  # 🚩只保留数值列
y = df[target]

# =========================
# 3. 计算相关系数矩阵
# =========================
corr_matrix = X.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.title("特征相关系数矩阵", fontsize=14)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()

# =========================
# 4. 计算 VIF
# =========================
X_with_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_with_const.values, i)
    for i in range(X_with_const.shape[1])
]

# 去掉常数项
vif_data = vif_data[vif_data["feature"] != "const"]

print("\n📊 多重共线性诊断（VIF 值）：")
print(vif_data.sort_values("VIF", ascending=False))
