import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 读取数据
file_path = "附件/附件4.xlsx"   # 修改为实际路径
df = pd.read_excel(file_path)

# 2. 删除无用列（姓名）
if "姓名" in df.columns:
    df = df.drop(columns=["姓名"])

# 3. 划分目标和特征
target = "成绩"
X = df.drop(columns=[target])
y = df[target]

# 4. 编码非数值型变量
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# 5. 标准化（可选，树模型不依赖，但相关性需要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=df.drop(columns=[target]).columns)

# 6. 皮尔逊相关系数
corr = pd.concat([X, y], axis=1).corr()[target].drop(target).sort_values(ascending=False)
print("📊 皮尔逊相关系数：")
print(corr)

# 7. 随机森林特征重要性
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    "特征": X.columns,
    "重要性": importances
}).sort_values("重要性", ascending=False)

print("\n🌲 随机森林特征重要性：")
print(importance_df)

# 8. 可视化
plt.figure(figsize=(12,6))
plt.bar(importance_df["特征"], importance_df["重要性"], color='skyblue')
plt.title("特征重要性 (随机森林)", fontsize=16)
plt.ylabel("重要性", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()
