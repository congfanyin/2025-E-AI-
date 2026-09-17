import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# 中文显示设置
# -------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 特征列
# -------------------------------
base_feature_names = [
    '下蹲髋膝踝', '下蹲肩髋膝', '下蹲脚踝膝', '下蹲腕肩髋',
    '起跳髋膝踝', '起跳肩髋膝', '起跳脚踝膝', '起跳腕肩髋', '躯干倾斜',
    '骨骼肌重量 (kg)', '基础代谢(kcal)', '肌肉率 (%)', '去脂体重 (kg)',
    '肌肉重量(kg)'
]

# -------------------------------
# 模型文件
# -------------------------------
compare_models = ["lasso_优化集成.pkl", "ridge_优化集成.pkl"]

# -------------------------------
# 示例运动员数据（每行对应一名运动员）
# -------------------------------
athletes_data = [
    [97.3, 61.64, 93.97, 13.67, 139.61, 151.95, 102.43, 161.76, 41.16,
     14.2, 1023, 50.9, 26.3, 15],
    [154.37,124.24,108.13,34.22,142.44,160.19,97.56,133.41,32.53,
     25.4,1260,48.3,38.9,22.3],
    [119.76,104.4,84.72,88.9,155.87,173.62,108.55,136.23,24.01,
     37.4,1678,51.8,56.1,34.7]
]

# 实际成绩（单位：米）
actual_results = [1.259, 1.33, 2.05]

# -------------------------------
# 转 DataFrame
# -------------------------------
X = pd.DataFrame(athletes_data, columns=base_feature_names)

# -------------------------------
# 结果表格
# -------------------------------
results = pd.DataFrame({"运动员": ["运动员11", "运动员3", "运动员5"]})
results["实际成绩(m)"] = actual_results

# 循环加载模型预测
for file in compare_models:
    model_name = file.replace(".pkl","")
    model = joblib.load(file)
    preds = model.predict(X)
    results[f"{model_name}_预测成绩(m)"] = preds
    results[f"{model_name}_误差(m)"] = preds - actual_results
    results[f"{model_name}_误差比例"] = (preds - actual_results) / actual_results

# -------------------------------
# 打印表格
# -------------------------------
print("模型预测与误差对比表：")
print(results.round(3))

# -------------------------------
# 可视化表格
# -------------------------------
fig, ax = plt.subplots(figsize=(14, 0.6 + len(X) * 0.5))
ax.axis('off')
table = ax.table(cellText=results.round(3).values,
                 colLabels=results.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()

# -------------------------------
# 条形图对比误差
# -------------------------------
athletes = results["运动员"].tolist()
x = range(len(athletes))
width = 0.35

ridge_errors = results["ridge_优化集成_误差(m)"]
lasso_errors = results["lasso_优化集成_误差(m)"]

fig, ax = plt.subplots(figsize=(10,5))
ax.bar([i - width/2 for i in x], ridge_errors, width=width, color='skyblue', label='Ridge误差')
ax.bar([i + width/2 for i in x], lasso_errors, width=width, color='salmon', label='Lasso误差')

ax.set_xticks(x)
ax.set_xticklabels(athletes)
ax.set_ylabel("预测误差 (m)")
ax.set_title("Ridge 与 Lasso 模型预测误差对比")
ax.legend()
plt.show()

# -------------------------------
# 折线图对比误差
# -------------------------------
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(athletes, ridge_errors, marker='o', color='skyblue', label='Ridge误差')
ax.plot(athletes, lasso_errors, marker='s', color='salmon', label='Lasso误差')
ax.axhline(0, color='gray', linestyle='--')  # 零误差线
ax.set_ylabel("预测误差 (m)")
ax.set_title("Ridge 与 Lasso 模型预测误差折线对比")
ax.legend()
plt.show()
