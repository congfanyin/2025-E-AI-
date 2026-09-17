import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
df = pd.read_excel("副本跳远角度汇总(1).xlsx")
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
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2.2 构造组合特征
if all(c in X_imputed.columns for c in ["髋膝踝", "肩髋膝"]):
    X_imputed["角度比值_髋膝踝/肩髋膝"] = X_imputed["髋膝踝"] / (X_imputed["肩髋膝"] + 1e-6)

if all(c in X_imputed.columns for c in ["肌肉重量(kg)", "去脂体重(kg)"]):
    X_imputed["肌肉/体重比"] = X_imputed["肌肉重量(kg)"] / (X_imputed["去脂体重(kg)"] + 1e-6)

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
    scores = cross_val_score(model, X_imputed, y, cv=kf, scoring="r2")
    print(f"{name}: 平均R²={np.mean(scores):.3f}, 标准差={np.std(scores):.3f}")

# =========================
# 5. 模型集成
# =========================
voting = VotingRegressor(estimators=base_models)
stacking = StackingRegressor(estimators=base_models, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]))

# =========================
# 6. 训练集/测试集划分
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 6.1 训练基础模型
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"{name} 已训练完成")

# 6.2 训练集成模型
voting.fit(X_train, y_train)
stacking.fit(X_train, y_train)

print("\n✅ VotingRegressor 测试集 R²:", voting.score(X_test, y_test))
print("✅ StackingRegressor 测试集 R²:", stacking.score(X_test, y_test))

# =========================
# 7. 保存所有模型
# =========================
all_models = base_models + [("voting", voting), ("stacking", stacking)]
for name, model in all_models:
    filename = f"{name}_优化集成.pkl"
    joblib.dump(model, filename)
    print(f"{name} 已保存为 {filename}")
