import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.impute import SimpleImputer

# XGBoost / LightGBM / CatBoost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("副本跳远角度汇总1（改）.xlsx")
df.columns = df.columns.str.strip()
target = "成绩"

# =========================
# 2. 特征选择与增强
# =========================
base_features = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]
X = df[base_features].copy()
y = df[target].copy()

# =========================
# 2.1 最小偏置
# =========================
y_shift = max(0.1 - y.min(), 0)
y += y_shift

# =========================
# 2.2 特征工程（组合特征）
# =========================
X["发力膝/髋比"] = X["发力膝角"] / (X["发力髋角"] + 1e-6)
X["起跳膝/髋比"] = X["起跳膝角"] / (X["起跳髋角"] + 1e-6)
X["肌肉/体重比"] = X["肌肉重量(kg)"] / (X["去脂体重 (kg)"] + 1e-6)
X["肌肉率×基础代谢"] = X["肌肉率 (%)"] * X["基础代谢(kcal)"]

# =========================
# 2.3 缺失值处理
# =========================
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# =========================
# 3. 基础模型
# =========================
base_models = [
    ("ols", Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])),
    ("ridge", Pipeline([("scaler", StandardScaler()), ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))])),
    ("lasso", Pipeline([("scaler", StandardScaler()), ("reg", LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000))])),
    ("rf", RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=3, random_state=42)),
    ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, min_samples_leaf=3, random_state=42)),
    ("xgb", XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)),
    ("lgb", LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42)),
    ("cat", CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=6, verbose=0, random_state=42))
]

# =========================
# 4. 交叉验证
# =========================
rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
print("📊 各模型交叉验证 R²：")
for name, model in base_models:
    scores = cross_val_score(model, X_imputed, y, cv=rkf, scoring="r2", n_jobs=-1)
    print(f"{name}: 平均R²={np.mean(scores):.3f}, 标准差={np.std(scores):.3f}")

# =========================
# 5. 集成模型
# =========================
voting = VotingRegressor(estimators=base_models)
stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-3,3,20), cv=5)
)

# =========================
# 6. 训练集/测试集划分
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# =========================
# 7. 训练基础模型并保存
# =========================
for name, model in base_models:
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_增强集成.pkl")
    print(f"{name} 已保存")

# =========================
# 8. 训练集成模型并保存
# =========================
voting.fit(X_train, y_train)
stacking.fit(X_train, y_train)
joblib.dump(voting, "voting_增强集成.pkl")
joblib.dump(stacking, "stacking_增强集成.pkl")
print("Voting 和 Stacking 已保存")

# =========================
# 9. 保存特征列和 imputer
# =========================
joblib.dump(X_imputed.columns.tolist(), "feature_columns_增强.pkl")
joblib.dump(imputer, "imputer_增强.pkl")
print("✅ 已保存特征列和 imputer")

# =========================
# 10. 测试集 R²
# =========================
print("\n✅ VotingRegressor 测试集 R²:", voting.score(X_test, y_test))
print("✅ StackingRegressor 测试集 R²:", stacking.score(X_test, y_test))
