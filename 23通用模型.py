import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RepeatedKFold

# ===============================
# 1. 导入数据
# ===============================
df = pd.read_excel("副本跳远角度汇总1（改）_扩展.xlsx")

# ===============================
# 2. 特征与目标
# ===============================
base_features = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]
target = '成绩'

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
# 5. 岭回归训练（10轮交叉验证）
# ===============================
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
alphas = np.logspace(-3, 3, 13)

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=alphas, cv=cv))
])

ridge_pipeline.fit(X_train, y_train)

# 保存模型
joblib.dump(ridge_pipeline, "岭回归.pkl")
print("✅ 岭回归模型已保存为 岭回归.pkl")

# ===============================
# 6. 检测预测
# ===============================
pred_train = ridge_pipeline.predict(X_train)
pred_test = ridge_pipeline.predict(X_test)

print(f"\n🏃‍♂️ 训练集预测R²: {ridge_pipeline.score(X_train, y_train):.3f}")
print(f"🏃‍♂️ 测试集预测R²: {ridge_pipeline.score(X_test, y_test):.3f}")

# ===============================
# 7. 新运动员预测示例
# ===============================
new_athlete = [[97.76, 60.38, 89.42, 9.96, 147.84, 165.81, 128.68, 138.62, 35.93,
                14.2, 1023, 50.9, 26.3, 15]]

X_new = pd.DataFrame(new_athlete, columns=base_features)
X_new_imputed = pd.DataFrame(imputer.transform(X_new), columns=X_new.columns)
pred_new = ridge_pipeline.predict(X_new_imputed)[0]

print(f"\n🏃‍♂️ 运动者11预测跳远成绩: {pred_new:.2f} m")
