import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimHei")

# ======================
# 1. 读取数据
# ======================
df = pd.read_excel("跳远角度汇总1.xlsx")
df.columns = df.columns.str.strip()

target = "成绩"

# 取数值型特征，排除目标列
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
features = [col for col in numeric_cols if col != target]

X = df[features].dropna()
y = df[target].loc[X.index]

# ======================
# 2. 定义 Ridge 回归模型（带标准化 Pipeline）
# ======================
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

# ======================
# 3. 交叉验证评估模型
# ======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(ridge_model, X, y, cv=kf, scoring="r2")

print(f"Ridge 回归模型交叉验证结果:")
print(f"平均 R²: {np.mean(scores):.3f}, 波动: {np.std(scores):.3f}")
print(f"每折 R²: {scores}")

# ======================
# 4. 拟合整个数据集
# ======================
ridge_model.fit(X, y)

# 输出回归系数
coef = ridge_model.named_steps["reg"].coef_
coef_dict = dict(zip(features, coef))
print("\n📌 Ridge 回归系数（标准化后）：")
for feat, weight in coef_dict.items():
    print(f"  {feat}: {weight:.4f}")

# ======================
# 5. 保存模型
# ======================
model_file = "ridge_model.pkl"
joblib.dump(ridge_model, model_file)
print(f"\n✅ Ridge 模型已保存至 {model_file}")
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# ======================
# 6. 可视化回归系数（归一化处理）
# ======================
coef_series = pd.Series(coef_dict).sort_values()

# 归一化处理（绝对值最大缩放到 1）
coef_norm = coef_series / coef_series.abs().max()

plt.figure(figsize=(10, 6))
sns.barplot(x=coef_norm.values, y=coef_norm.index, palette="coolwarm")

plt.axvline(0, color="black", linewidth=1)  # 参考线
plt.title("Ridge 回归系数可视化（归一化后）", fontsize=14, weight="bold")
plt.xlabel("归一化系数 (相对强度)", fontsize=12)
plt.ylabel("特征", fontsize=12)

# 在柱子上标注原始系数数值
for i, v in enumerate(coef_norm.values):
    raw_val = coef_series.values[i]  # 原始系数
    plt.text(v, i, f"{raw_val:.2f}", va="center",
             ha="left" if v > 0 else "right",
             fontsize=9, color="black")

plt.tight_layout()
plt.show()
