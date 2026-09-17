import pandas as pd
import joblib

# 模型使用的特征列（训练时）
base_feature_names = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]

model_files = [

    "岭回归.pkl",
]

def predict_athletes(athletes_data):
    # ⚠️ 这里直接用完整列名，不要去掉
    X = pd.DataFrame(athletes_data, columns=base_feature_names)

    # 循环加载模型预测
    for file in model_files:
        model = joblib.load(file)
        predictions = model.predict(X)
        model_name = file.replace(".pkl", "")
        for i, pred in enumerate(predictions):
            print(f"{model_name} - 运动员{i + 1}预测跳远成绩 (m)：{pred:.2f}")

# =========================
# 示例使用
# =========================
athletes_data = [
   [97.76, 60.38, 89.42, 9.96, 147.84, 165.81, 128.68, 138.62, 35.93,
     14.2, 1023, 50.9, 26.3, 15]]# 14列，对应base_feature_names ]

predict_athletes(athletes_data)
