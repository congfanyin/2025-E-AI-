
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======================
# 1. 设置中文字体 & 美化风格
# ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimHei")

# =========================
# 1. 导入运动员测试数据
# =========================
from shuju import athletes_data  # 你的测试数据

# 第一行是列名，剩下的是数据
columns = athletes_data[0]
data = athletes_data[1:]
X_new = pd.DataFrame(data, columns=columns)

# =========================
# 2. 基础特征列（训练时使用的列）
# =========================
base_feature_names = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]

# =========================
# 3. 检查缺失列
# =========================
missing_cols = [col for col in base_feature_names if col not in X_new.columns]
if missing_cols:
    print(f"❌ 测试数据缺少以下必要特征列: {missing_cols}")
    raise ValueError("请检查测试数据，缺少必需特征列。")

X_new = X_new[base_feature_names]

# 后续步骤不变


# =========================
# 4. 缺失值处理
# =========================
imputer = SimpleImputer(strategy="median")
X_new_imputed = pd.DataFrame(imputer.fit_transform(X_new), columns=X_new.columns)

# =========================
# 5. 构造组合特征
# =========================
X_new_imputed["发力膝角/肩髋膝比值"] = X_new_imputed["发力膝角"] / (X_new_imputed["起跳髋角"] + 1e-6)
X_new_imputed["起跳膝角/肩髋膝比值"] = X_new_imputed["起跳膝角"] / (X_new_imputed["起跳髋角"] + 1e-6)
X_new_imputed["肌肉/体重比"] = X_new_imputed["肌肉重量(kg)"] / (X_new_imputed["去脂体重 (kg)"] + 1e-6)

# =========================
# 6. 对齐训练列顺序（这里假设和训练时一致）
# =========================
trained_feature_names = list(X_new_imputed.columns)
X_new_aligned = X_new_imputed[trained_feature_names]

# =========================
# 7. 加载训练好的模型
# =========================
ridge_model = joblib.load("ridge_优化集成.pkl")

# =========================
# 8. 模型预测 + 调试信息
# =========================
predictions = ridge_model.predict(X_new_aligned)
print("🏃‍♂️ 预测成绩：")
for i, pred in enumerate(predictions):
    print(f"运动员 {i+1} 预测成绩: {pred:.2f}")

# =========================
# 9. 分析线性模型系数对角度影响
# =========================
if hasattr(ridge_model, "named_steps"):  # Pipeline
    reg = ridge_model.named_steps["reg"]
else:
    reg = ridge_model

coefficients = reg.coef_
feature_names = X_new_aligned.columns

angles_of_interest = [

    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜'
]


print("\n📊 各角度影响分析：")
for angle in angles_of_interest:
    if angle in feature_names:
        idx = feature_names.get_loc(angle)
        coef = coefficients[idx]
        direction = "正向" if coef > 0 else "负向"
        print(f"{angle}: 系数={coef:.4f}, {direction}影响")

# 找出影响最大的角度
abs_coef = np.abs(coefficients)
max_idx = np.argmax(abs_coef)
print(f"\n🔥 对成绩影响最深的角度: {feature_names[max_idx]}, 系数={coefficients[max_idx]:.4f}")


# =========================
# 10. 各部位最佳角度范围参考（最近5000条，均值 ±2°）
# =========================
print("\n🏆 各部位最佳角度范围参考（预测成绩最高的1000条，均值 ±1°）：")

# =========================
# 1. 先预测全量数据成绩
# =========================
X_all_aligned = X_new_aligned[trained_feature_names]  # 对齐训练列顺序
all_predictions = ridge_model.predict(X_all_aligned)  # 全量预测

# =========================
# 2. 取预测成绩最高的1000条
# =========================
top_idx = all_predictions.argsort()[-1000:]  # 最大的1000条索引
X_top1000 = X_new_aligned.iloc[top_idx]

# =========================
# 3. 输出各部位角度均值范围（±1°）
# =========================
for angle in angles_of_interest:
    if angle in X_top1000.columns:
        mean_val = X_top1000[angle].mean()
        best_range = (mean_val - 1, mean_val + 1)
        print(f"{angle}: {best_range[0]:.2f} ~ {best_range[1]:.2f}")

# =========================
# 4. 构建均值向量（用于预测）
# =========================
mean_features = X_top1000.mean()  # 每列均值
X_mean_vector = mean_features.to_frame().T  # 转成单行 DataFrame

# 组合特征（与训练时一致）
X_new_imputed["发力膝角/肩髋膝比值"] = X_new_imputed["发力膝角"] / (X_new_imputed["起跳髋角"] + 1e-6)
X_new_imputed["起跳膝角/肩髋膝比值"] = X_new_imputed["起跳膝角"] / (X_new_imputed["起跳髋角"] + 1e-6)
X_new_imputed["肌肉/体重比"] = X_new_imputed["肌肉重量(kg)"] / (X_new_imputed["去脂体重 (kg)"] + 1e-6)

# =========================
# 5. 对齐训练列顺序
# =========================
X_mean_aligned = X_mean_vector[trained_feature_names]

# =========================
# 6. 使用均值数据预测成绩
# =========================
mean_prediction = ridge_model.predict(X_mean_aligned)[0]
print(f"\n📈 使用预测成绩最高的1000条均值角度数据预测成绩: {mean_prediction:.2f}")
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 假设有一列能标记运动员项目，比如 "项目"
event_col = "项目"

# =========================
# 只保留跳远者的数据
# =========================
import matplotlib.pyplot as plt

# =========================
# 按成绩排序
# =========================
sorted_predictions = np.sort(all_predictions)  # 从低到高排序
x_vals = np.arange(1, len(sorted_predictions) + 1)  # 从 1 到 N

# =========================
# 绘制曲线
# =========================
plt.figure(figsize=(12, 6))
plt.plot(x_vals, sorted_predictions, label="预测成绩走势", color="blue")

# 高亮前 1000 条最佳
plt.axvline(len(sorted_predictions) - 1000, color="red", linestyle="--", label="Top 1000 分界")

plt.xlabel("运动员排序（次数）")
plt.ylabel("预测成绩")
plt.title("全部运动员预测成绩分布（按从低到高排序）")
plt.legend()
plt.show()
