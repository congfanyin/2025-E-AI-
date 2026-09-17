import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.cluster import KMeans

# ===============================
# 1. 导入数据
# ===============================
from shuju import athletes_data

columns = athletes_data[0]
data = athletes_data[1:]
df = pd.DataFrame(data, columns=columns)

# ===============================
# 2. 特征与目标
# ===============================
angles_of_interest = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜'
]

base_features = angles_of_interest + [
    '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]

target = '成绩'

missing_cols = [col for col in base_features+[target] if col not in df.columns]
if missing_cols:
    raise ValueError(f"数据缺少必要列: {missing_cols}")

X = df[base_features]
y = df[target]

# ===============================
# 3. 缺失值处理
# ===============================
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ===============================
# 4. 数据划分
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ===============================
# 5. 岭回归训练 (10轮交叉验证)
# ===============================
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
alphas = np.logspace(-3, 3, 13)

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=alphas, cv=cv))
])

ridge_pipeline.fit(X_train, y_train)
joblib.dump(ridge_pipeline, "岭回归.pkl")
print("✅ 岭回归模型已保存为 岭回归.pkl")

# ===============================
# 6. 对新运动员进行预测
# ===============================
new_athlete = [
#     [97.76, 60.38, 89.42, 9.96, 147.84, 165.81, 128.68, 138.62, 35.93,
#      14.2, 1023, 50.9, 26.3, 15]
# ]
[128.96, 65.27, 82.67, 60.15, 176.2, 165.23, 142.7, 134.95, 46.38,
     14.2, 1023, 50.9, 26.3, 15]
]
X_new = pd.DataFrame(new_athlete, columns=base_features)
X_new_imputed = pd.DataFrame(imputer.transform(X_new), columns=X_new.columns)
X_new_aligned = X_new_imputed[X_train.columns]
pred_new = ridge_pipeline.predict(X_new_aligned)[0]
print(f"\n🏃‍♂️ 运动者11预测跳远成绩: {pred_new:.2f} m")

# ===============================
# 7. 聚类分析提取最佳角度
# ===============================
# 标准化训练数据用于聚类
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# KMeans聚类，假设分为3类
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
X_imputed['Cluster'] = clusters

# 找出预测成绩最高的群体
all_predictions = ridge_pipeline.predict(X_imputed[X_train.columns])
X_imputed['Predicted'] = all_predictions
top_30pct_threshold = np.percentile(all_predictions, 70)
best_group = X_imputed[X_imputed['Predicted'] >= top_30pct_threshold]

# 计算各角度均值（即最佳角度）
best_angles = best_group[angles_of_interest].mean()
best_angles_list = best_angles.tolist()

print("\n🔥 对成绩正向提升最大的各角度（聚类分析+前30%成绩均值）：")
for angle, val in zip(angles_of_interest, best_angles_list):
    print(f"{angle}: {val:.2f}")
