import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RepeatedKFold

# ===============================
# 1. 导入数据
# ===============================
from shuju import athletes_data  # 数据源

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

body_info_features = [
    '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]

base_features = angles_of_interest + body_info_features
target = '成绩'

# 检查缺失列
missing_cols = [col for col in base_features + [target] if col not in df.columns]
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
# 5. RepeatedKFold 10轮交叉验证 + 岭回归训练
# ===============================
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
alphas = np.logspace(-3, 3, 13)

ridge_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('ridge', RidgeCV(alphas=alphas, cv=cv))
])

ridge_pipeline.fit(X_train, y_train)

# 保存模型
joblib.dump(ridge_pipeline, "岭回归.pkl")
print("✅ 岭回归模型已保存为 岭回归.pkl")

# ===============================
# 6. 分析角度影响系数
# ===============================
ridge_model = ridge_pipeline.named_steps['ridge']
coef = ridge_model.coef_
feature_names = X_train.columns

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
angles_coef = coef_df[coef_df['Feature'].isin(angles_of_interest)].copy()
angles_coef['Abs_Coefficient'] = angles_coef['Coefficient'].abs()
angles_coef = angles_coef.sort_values(by='Abs_Coefficient', ascending=False)

print("\n📊 各角度影响系数：")
print(angles_coef[['Feature', 'Coefficient']])

max_angle = angles_coef.iloc[0]
print(f"\n🔥 对成绩影响最大的角度: {max_angle['Feature']}, 系数={max_angle['Coefficient']:.4f}")

# ===============================
# 7. 使用新运动员数据预测
# ===============================
new_athlete = [
    [97.76, 60.38, 89.42, 9.96, 147.84, 165.81, 128.68, 138.62, 35.93,
     14.2, 1023, 50.9, 26.3, 15]
]

X_new = pd.DataFrame(new_athlete, columns=base_features)
X_new_imputed = pd.DataFrame(imputer.transform(X_new), columns=X_new.columns)

# 对齐列顺序并预测
X_new_aligned = X_new_imputed[feature_names]
pred = ridge_pipeline.predict(X_new_aligned)[0]

# ===============================
# 8. 前30%运动员最佳角度分析
# ===============================
# 对训练集进行预测
X_train_aligned = X_train[feature_names]
y_train_pred = ridge_pipeline.predict(X_train_aligned)

df_pred = X_train.copy()
df_pred['预测成绩'] = y_train_pred

# 取前30%
top30_count = int(len(df_pred) * 0.3)
df_top30 = df_pred.nlargest(top30_count, '预测成绩')

# 最佳角度均值
best_angles_top30 = [df_top30[angle].mean() for angle in angles_of_interest]

# 后面5个体质信息均值
body_info_means = [X_train[f].mean() for f in body_info_features]

# 构建完整列表
best_angles_list = best_angles_top30 + body_info_means

print("\n🏆 前30%运动员最佳角度 + 体质信息列表：")

print(f"\n🏃‍♂️运动者11的基础预测跳远成绩: {pred:.2f} m")
import pandas as pd

# 前9个最佳角度
best_angles_list = [124.13, 63.85, 79.00, 58.85, 164.12, 169.03, 141.50, 133.92, 41.57]

# 后5个体质信息
athlete_extra = [14.2, 1023, 50.9, 26.3, 15]

# 组合成完整特征向量
full_features = best_angles_list + athlete_extra

# 训练时的列顺序
feature_columns = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]

# 构建 DataFrame
X_test = pd.DataFrame([full_features], columns=feature_columns)

# 使用训练好的岭回归模型预测
predicted_score = ridge_pipeline.predict(X_test)[0]
print(f"🏃‍♂️ 运动者11理想最佳跳远成绩: {predicted_score:.2f} m")

