import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. 设置中文字体 & 美化风格
# ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="SimHei")

# ======================
# 2. 读取数据
# ======================
df = pd.read_excel("跳远角度汇总1.xlsx")

target = "成绩"
features = '下蹲髋膝踝', '下蹲肩髋膝', '下蹲脚踝膝', '下蹲腕肩髋','起跳髋膝踝', '起跳肩髋膝', '起跳脚踝膝', '起跳腕肩髋', '躯干倾斜','骨骼肌重量 (kg)', '基础代谢(kcal)', '肌肉率 (%)', '去脂体重 (kg)','肌肉重量(kg)'
# features = "髋膝踝", "肩髋膝", "腕摆动", '脚踝膝', "躯干倾斜", "骨骼肌重量 (kg)", "体脂率 (%)", "肌肉率 (%)", '去脂体重 (kg)'
# features = (
#     "髋膝踝", "肩髋膝", "腕摆动", "脚踝膝", "躯干倾斜",
#     "骨骼肌重量 (kg)", "脂肪控制量 (kg)", "身高 (cm)", "肌肉重量 (kg)", "标准体重 (kg)"
# )

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=[target] + list(features))

# ======================
# 3. 计算相关性
# ======================
results = []
for col in features:
    x = df[target].astype(float)
    y = df[col].astype(float)

    mask = x.notna() & y.notna()
    x_valid, y_valid = x[mask], y[mask]

    if len(x_valid) < 2:
        continue

    r, p = pearsonr(x_valid, y_valid)
    rho, p_s = spearmanr(x_valid, y_valid)

    results.append({
        "变量": col,
        "Pearson r": round(r, 6),
        "Pearson p值": p,
        "Spearman ρ": round(rho, 6),
        "Spearman p值": p_s
    })

result_df = pd.DataFrame(results)

# ======================
# 4. 绘制条形图（p值）
# ======================
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = range(len(result_df))

def get_color_and_height(p):
    """根据p值决定颜色和显示高度"""
    if p < 0.01:
        return "green", 0.01, None
    elif p < 0.05:
        return "red", p, None
    else:
        return "blue", p, None

# Pearson 柱子
for i, p in enumerate(result_df["Pearson p值"]):
    color, height, label = get_color_and_height(p)
    plt.bar(i - bar_width/2, height, width=bar_width, color=color, label=None)
    # 添加标注
    if label:
        plt.text(i - bar_width/2, height + 0.005, label, ha="center", va="bottom", fontsize=9)
    else:
        plt.text(i - bar_width/2, height + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=9)

# Spearman 柱子
for i, p in enumerate(result_df["Spearman p值"]):
    color, height, label = get_color_and_height(p)
    plt.bar(i + bar_width/2, height, width=bar_width, color=color, hatch="//", label=None)
    # 添加标注
    if label:
        plt.text(i + bar_width/2, height + 0.005, label, ha="center", va="bottom", fontsize=9)
    else:
        plt.text(i + bar_width/2, height + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=9)

# 添加显著性阈值线
plt.axhline(y=0.05, color="black", linestyle="--", linewidth=1.5, label="显著性阈值 0.05")

# 美化图表
plt.xticks(x, result_df["变量"], rotation=45, ha="right", fontsize=11)
plt.ylabel("显著性 p值", fontsize=12)
plt.title("调整前后跳远成绩与特征的相关性显著性检验", fontsize=14, weight="bold")

# 添加自定义图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green", label="特别显著 (p < 0.01)"),
    Patch(facecolor="red", label="显著 (p < 0.05)"),
    Patch(facecolor="blue", label="不显著 (p ≥ 0.05)"),
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.show()
