import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("副本跳远角度汇总1（改）.xlsx")
df.columns = df.columns.str.strip()
target = "成绩"

# =========================
# 2. 特征选择和特征工程
# =========================
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
features = [col for col in numeric_cols if col != target]

X = df[features]
y = df[target]

# 2.1 缺失值填充
imputer = SimpleImputer(strategy="median")
X_new_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2.2 构造组合特征（角度比值、肌肉/体重比等）
# 假设角度列为 A1, A2, A3，肌肉重量列为 Muscle，体重列为 Weight
# 请根据你的实际列名修改
X_new_imputed["发力膝角/肩髋膝比值"] = X_new_imputed["发力膝角"] / (X_new_imputed["起跳髋角"] + 1e-6)
X_new_imputed["起跳膝角/肩髋膝比值"] = X_new_imputed["起跳膝角"] / (X_new_imputed["起跳髋角"] + 1e-6)
X_new_imputed["肌肉/体重比"] = X_new_imputed["肌肉重量(kg)"] / (X_new_imputed["去脂体重 (kg)"] + 1e-6)


# =========================
# 3. 定义基础模型
# =========================
base_models = [
    ("ols", Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])),
    ("ridge", Pipeline([("scaler", StandardScaler()), ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))])),
    ("lasso", Pipeline([("scaler", StandardScaler()), ("reg", LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000))])),
    ("rf", RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=3, random_state=42)),
    ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, min_samples_leaf=3, random_state=42))
]

# =========================
# 4. K折交叉验证评估
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("📊 各模型交叉验证 R²：")
for name, model in base_models:
    scores = cross_val_score(model, X_new_imputed, y, cv=kf, scoring="r2")
    print(f"{name}: 平均R²={np.mean(scores):.3f}, 标准差={np.std(scores):.3f}")

# =========================
# 5. 模型集成
# =========================
# 5.1 VotingRegressor (简单平均)
voting = VotingRegressor(estimators=base_models)
# 5.2 StackingRegressor (以线性模型为元模型)
stacking = StackingRegressor(estimators=base_models, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]))

# =========================
# 6. 训练集/测试集划分
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_new_imputed, y, test_size=0.2, random_state=42)

voting.fit(X_train, y_train)
stacking.fit(X_train, y_train)

print("\n✅ VotingRegressor 测试集 R²:", voting.score(X_test, y_test))
print("✅ StackingRegressor 测试集 R²:", stacking.score(X_test, y_test))

# =========================
# 6. 训练基础模型并保存
# =========================
for name, model in base_models:
    model.fit(X_train, y_train)  # ⚠️ 确保每个基础模型都训练过
    filename = f"{name}_优化集成.pkl"
    joblib.dump(model, filename)
    print(f"{name} 已保存为 {filename}")

# =========================
# 7. 训练集成模型并保存
# =========================
voting.fit(X_train, y_train)
stacking.fit(X_train, y_train)

for name, model in [("voting", voting), ("stacking", stacking)]:
    filename = f"{name}_优化集成.pkl"
    joblib.dump(model, filename)
    print(f"{name} 已保存为 {filename}")
