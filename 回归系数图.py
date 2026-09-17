import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======================
# 1. 设置中文字体 & 美化风格
# ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimHei")

# -------------------------------
# 1. 加载 Ridge 模型
# -------------------------------
model_file = "ridge_优化集成.pkl"
ridge_model = joblib.load(model_file)

# 如果是 Pipeline，需要先提取 RidgeCV 模型
if hasattr(ridge_model, 'named_steps'):
    reg = ridge_model.named_steps['reg']
else:
    reg = ridge_model

# -------------------------------
# 2. 获取特征列名（训练时用的）
# -------------------------------
feature_names = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]

# -------------------------------
# 3. 提取系数（保持原始值）
# -------------------------------
coef_dict = dict(zip(feature_names, reg.coef_))
coef_series = pd.Series(coef_dict).sort_values()

# -------------------------------
# 4. 绘制柱状图（不做归一化）
# -------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=coef_series.values, y=coef_series.index, palette="coolwarm")

plt.axvline(0, color="black", linewidth=1)  # 参考线
plt.title("Ridge 回归系数可视化（原始系数）", fontsize=14, weight="bold")
plt.xlabel("回归系数", fontsize=12)
plt.ylabel("特征", fontsize=12)

# 在柱子上标注原始系数数值
for i, v in enumerate(coef_series.values):
    plt.text(v, i, f"{v:.2f}", va="center",
             ha="left" if v > 0 else "right",
             fontsize=9, color="black")

plt.tight_layout()
plt.show()
