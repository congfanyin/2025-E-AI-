import pandas as pd
import numpy as np
import statistics as st

# -------------------------------
# 三点夹角计算函数
# -------------------------------
def angle_between_three_points(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# -------------------------------
# 向量夹角
# -------------------------------
def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# -------------------------------
# 起跳帧判定：第一个手臂角度 >100°，从100帧开始
# -------------------------------
def find_jump_frame(df):
    wrist_angles = []
    for i, row in df.iterrows():
        wrist = angle_between_three_points(
            (st.mean([row['19_X'], row['20_X']]), st.mean([row['19_Y'], row['20_Y']])),  # 手腕
            (st.mean([row['11_X'], row['12_X']]), st.mean([row['11_Y'], row['12_Y']])),  # 肩
            (st.mean([row['23_X'], row['24_X']]), st.mean([row['23_Y'], row['24_Y']]))   # 髋
        )
        wrist_angles.append(round(wrist, 2))

    df['wrist_angle'] = wrist_angles

    # 限制在 100帧之后
    jump_idx_candidates = df[(df['帧号'] >= 100) & (df['wrist_angle'] > 140)]
    if not jump_idx_candidates.empty:
        jump_idx = jump_idx_candidates.index[0]
        jump_frame = df.loc[jump_idx, '帧号']
        return jump_idx, jump_frame
    else:
        return None, None


# -------------------------------
# 落地帧判定（第二象限）
# -------------------------------
def find_landing_frame(df, jump_idx):
    if jump_idx is None or jump_idx <= 0:
        return None

    # 起跳帧前一帧脚跟坐标 = 直角点
    heel_pre = np.array([
        st.mean([df.loc[jump_idx-1, '29_X'], df.loc[jump_idx-1, '30_X']]),
        st.mean([df.loc[jump_idx-1, '29_Y'], df.loc[jump_idx-1, '30_Y']])
    ])

    # 构造参考向量
    x_axis = np.array([1, 0])   # 水平
    y_axis = np.array([0, 1])   # 竖直

    # 从起跳帧之后开始找
    for i in range(jump_idx+1, len(df)):
        heel = np.array([st.mean([df.loc[i, '29_X'], df.loc[i, '30_X']]),
                         st.mean([df.loc[i, '29_Y'], df.loc[i, '30_Y']])])
        toe = np.array([st.mean([df.loc[i, '31_X'], df.loc[i, '32_X']]),
                        st.mean([df.loc[i, '31_Y'], df.loc[i, '32_Y']])])

        # 脚跟 -> 脚尖向量
        foot_vec = toe - heel

        # 条件1：和直角两边比较，取最小角
        angle_to_x = angle_between_vectors(foot_vec, x_axis)
        angle_to_y = angle_between_vectors(foot_vec, y_axis)
        min_angle = min(angle_to_x, angle_to_y)

        # -------------------
        # 新条件：第二象限
        # -------------------
        rel_toe = toe - heel_pre  # 相对于起跳帧前一帧的脚跟
        in_quadrant_2 = (rel_toe[0] < 0) and (rel_toe[1] > 0)

        # 条件2：和x轴误差小于5 且 第二象限
        if (min_angle < 45) and (angle_to_x < 5) and in_quadrant_2:
            return df.loc[i, '帧号']

    return None


# -------------------------------
# 主流程
# -------------------------------
if __name__ == "__main__":
    df = pd.read_excel(r'附件/附件5/运动者11的跳远位置信息_修正.xlsx')

    # 起跳帧
    jump_idx, jump_frame = find_jump_frame(df)
    print("起跳帧:", jump_frame)

    # 落地帧
    land_frame = find_landing_frame(df, jump_idx)
    print("落地帧:", land_frame)
