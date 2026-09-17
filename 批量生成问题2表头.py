import re
import pandas as pd

def parse_score_file(txt_path):
    data = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    athlete = None
    for line in lines:
        # 去掉首尾空白，把制表符替换为普通空格
        line = line.strip().replace("\t", " ")
        # 连续空格压缩成一个
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue

        # 匹配运动者编号（如 "运动者7"）
        m_ath = re.match(r"运动者(\d+)", line)
        if m_ath:
            athlete = f"运动者{m_ath.group(1)}"
            continue

        # 匹配 “第n次 x米”，允许整数或小数
        m_score = re.match(r"第(\d+)次\s+([\d\.]+)米", line)
        if m_score and athlete is not None:
            attempt = int(m_score.group(1))
            score = float(m_score.group(2))
            data.append([athlete, attempt, score])

    return pd.DataFrame(data, columns=["运动员", "次数", "成绩"])


# =============== 使用示例 ==================
df_before_score = parse_score_file(r"D:\math\附件\附件3\姿势调整前\运动者姿势调整前的跳远成绩.txt")
df_after_score  = parse_score_file(r"D:\math\附件\附件3\姿势调整后\运动者姿势调整后的跳远成绩.txt")

# 合并并计算成绩差值
df_score = pd.merge(df_before_score, df_after_score,
                    on=["运动员", "次数"], suffixes=("_前", "_后"))

df_score["成绩差值"] = df_score["成绩_后"] - df_score["成绩_前"]

# 保存
df_score.to_excel("成绩差值表.xlsx", index=False)
print("✅ 成绩差值表已保存")
