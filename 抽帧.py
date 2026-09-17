import cv2
import os

# 视频路径
video_path = r"D:\math\附件\附件5\运动者11的跳远视频.mp4"
# 保存帧的文件夹
save_dir = "frames"

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频")
    exit()

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("视频总帧数:", total_frames)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 生成保存路径
    save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(save_path, frame)

    print(f"已保存: {save_path}")
    frame_idx += 1

cap.release()
print("所有帧保存完成 ✅")
