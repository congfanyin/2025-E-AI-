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
df = pd.read_excel("副本跳远角度汇总1（改）.xlsx")
df.columns = df.columns.str.strip()
target = "成绩"

# =========================
# 2. 特征选择和特征工程
# =========================
base_feature_names = [
    '发力膝角', '发力髋角', '发力踝角', '发力腕角',
    '起跳膝角', '起跳髋角', '起跳踝角', '起跳肩角',
    '躯干倾斜', '骨骼肌重量 (kg)', '基础代谢(kcal)',
    '肌肉率 (%)', '去脂体重 (kg)', '肌肉重量(kg)'
]
X = df[base_feature_names]
y = df[target].copy()

# =========================
# 2.1 对目标进行最小偏置，保证预测值 >=0.1
# =========================
y_shift = max(0.1 - y.min(), 0)
y = y + y_shift

# =========================
# 2.2 缺失值填充
# =========================
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# =========================
# 2.3 构造组合特征
# =========================
if all(c in X_imputed.columns for c in ["发力膝角", "发力髋角"]):
    X_imputed["发力膝角/肩髋膝比值"] = X_imputed["发力膝角"] / (X_imputed["发力髋角"] + 1e-6)
if all(c in X_imputed.columns for c in ["起跳膝角", "起跳髋角"]):
    X_imputed["起跳膝角/肩髋膝比值"] = X_imputed["起跳膝角"] / (X_imputed["起跳髋角"] + 1e-6)
if all(c in X_imputed.columns for c in ["肌肉重量(kg)", "去脂体重 (kg)"]):
    X_imputed["肌肉/体重比"] = X_imputed["肌肉重量(kg)"] / (X_imputed["去脂体重 (kg)"] + 1e-6)

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

# =========================
# 7. 训练基础模型并保存
# =========================
for name, model in base_models:
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_优化集成1.pkl")
    print(f"{name} 已保存")

# =========================
# 8. 训练集成模型并保存
# =========================
voting.fit(X_train, y_train)
stacking.fit(X_train, y_train)
for name, model in [("voting", voting), ("stacking", stacking)]:
    joblib.dump(model, f"{name}_优化集成.pkl")
    print(f"{name} 已保存")

# =========================
# 9. 保存训练特征列顺序和 imputer
# =========================
joblib.dump(X_imputed.columns.tolist(), "feature_columns.pkl")
joblib.dump(imputer, "imputer.pkl")
print("✅ 已保存训练特征列顺序和 imputer")

# =========================
# 10. 测试集 R²
# =========================
print("\n✅ VotingRegressor 测试集 R²:", voting.score(X_test, y_test))
print("✅ StackingRegressor 测试集 R²:", stacking.score(X_test, y_test))

# =========================
# 11. 岭回归专用输出（系数表 + 方差分析）
# =========================
ridge_model = [m for n, m in base_models if n == "ridge"][0]
ridge_model.fit(X_train, y_train)

if hasattr(ridge_model, "named_steps"):
    reg = ridge_model.named_steps["reg"]
    scaler = ridge_model.named_steps["scaler"]
    X_scaled = scaler.transform(X_train)
else:
    reg = ridge_model
    X_scaled = X_train.values

# 系数表
coef_df = pd.DataFrame({
    "Feature": X_imputed.columns,
    "Coefficient": reg.coef_
})
coef_df.loc[len(coef_df)] = ["Intercept", reg.intercept_]

# 方差分析表（近似版）
y_pred = ridge_model.predict(X_train)
residuals = y_train - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
r2 = 1 - ss_res / ss_tot

anova_df = pd.DataFrame({
    "Metric": ["样本数", "特征数", "SSE(残差平方和)", "SST(总平方和)", "R²(决定系数)"],
    "Value": [len(y_train), X_train.shape[1], ss_res, ss_tot, r2]
})

# 保存 Excel
with pd.ExcelWriter("岭回归模型结果.xlsx") as writer:
    coef_df.to_excel(writer, sheet_name="系数表", index=False)
    anova_df.to_excel(writer, sheet_name="方差分析表", index=False)

print("✅ 已生成 岭回归模型结果.xlsx (包含 系数表 + 方差分析表)")
