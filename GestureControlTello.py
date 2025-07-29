# ===============================================================
#  基于手势的 Tello EDU 无人机实时控制（480×320 预览窗口）
#  主要依赖：
#    - djitellopy        : 控制 Tello
#    - mediapipe         : 手部关键点检测
#    - opencv-python     : 图像显示与绘制
# ===============================================================

import cv2
import mediapipe as mp
from djitellopy import Tello
import threading
import queue
import time
import numpy as np


# ----------------------------------------------------------------
#  1. 线程化读取：把 OpenCV 的摄像头读取放在独立线程，
#     避免主线程阻塞，提高帧率。
# ----------------------------------------------------------------
class CamThread(threading.Thread):
    def __init__(self, cap, q):
        super().__init__()
        self.cap = cap          # cv2.VideoCapture 对象
        self.q   = q            # 线程安全的队列
        self.running = True     # 控制线程结束的标志

    def run(self):
        """持续读取摄像头帧，放入队列"""
        while self.running:
            ok, img = self.cap.read()
            if ok:
                # 如果队列已满，丢弃最老的一帧，保证实时性
                if self.q.full():
                    self.q.get_nowait()
                self.q.put(img)

    def stop(self):
        """主线程结束时调用，通知本线程退出"""
        self.running = False


# ----------------------------------------------------------------
#  2. 初始化
# ----------------------------------------------------------------
# 2-1 连接无人机
tello = Tello()
tello.connect()
print("电池电量：", tello.get_battery(), "%")
tello.send_rc_control(0, 0, 0, 0)   # 先全部清零，防止残留指令

# 2-2 打开本地摄像头（如果你想用 Tello 的图传，请改成 tello.get_frame_read()）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 120)

# 2-3 启动读取线程
q = queue.Queue(maxsize=2)          # 只缓存 2 帧，保证低延迟
cam_thread = CamThread(cap, q)
cam_thread.start()

# ----------------------------------------------------------------
#  3. 初始化 MediaPipe 手势识别
# ----------------------------------------------------------------
mpHands = mp.solutions.hands
mpDraw  = mp.solutions.drawing_utils
hands = mpHands.Hands(
    model_complexity=0,            # 轻量模型，速度更快
    max_num_hands=2,               # 最多检测两只手
    min_detection_confidence=0.75, # 检测阈值
    min_tracking_confidence=0.5    # 跟踪阈值
)

# ----------------------------------------------------------------
#  4. 界面几何参数（按 480×320 预览分辨率设计）
# ----------------------------------------------------------------
takeoff_btn = (300, 160)   # 起飞按钮圆心
takeoff_rad = 45           # 半径

left_js     = (150, 250)   # 左“摇杆”圆心（控制前后左右）
left_rad    = 40

right_js    = (400, 250)   # 右“摇杆”圆心（控制上升/下降）
right_rad   = 40

land_btn    = (300, 160)   # 降落按钮圆心
land_rad    = 35

dead_zone   = 6            # 摇杆死区（像素）
speed       = 30           # 默认速度 (cm/s)
flying      = False        # 起飞状态标志


# ----------------------------------------------------------------
#  5. 工具函数
# ----------------------------------------------------------------
def dist(p1, p2):
    """计算两点欧氏距离"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ----------------------------------------------------------------
#  6. 主循环
# ----------------------------------------------------------------
while True:
    # 6-1 取一帧
    if q.empty():
        time.sleep(0.001)   # 没有帧时短暂挂起
        continue
    img = cv2.flip(q.get(), 1)   # 水平镜像，让“左右”符合直觉
    h, w = img.shape[:2]

    # 6-2 手势识别
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # 6-3 绘制 UI
    if not flying:
        # 未起飞：只画绿色“TAKE OFF”按钮
        cv2.circle(img, takeoff_btn, takeoff_rad, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(img, "TAKE OFF", (takeoff_btn[0] - 55, takeoff_btn[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        # 已起飞：画两个白色摇杆 + 红色降落按钮
        cv2.circle(img, left_js,   left_rad,   (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, right_js,  right_rad,  (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, land_btn,  land_rad,   (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "LAND", (land_btn[0] - 25, land_btn[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 6-4 处理检测到的手
    if results.multi_hand_landmarks:
        # 只取第一只手
        hand = results.multi_hand_landmarks[0]
        # 画骨架
        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        # 取食指指尖关键点 (landmark[8])
        idx = hand.landmark[8]
        x, y = int(idx.x * w), int(idx.y * h)
        cv2.circle(img, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)

        # 6-4-1 起飞检测
        if not flying and dist((x, y), takeoff_btn) < takeoff_rad:
            tello.takeoff()
            flying = True
            print("已起飞")

        # 6-4-2 起飞后的操作
        if flying:
            # 降落检测
            if dist((x, y), land_btn) < land_rad:
                print("降落中...")
                tello.land()
                break   # 直接跳出主循环，进入清理阶段

            # 左摇杆：前后左右
            if dist((x, y), left_js) < left_rad:
                dx, dy = x - left_js[0], y - left_js[1]
                vx = vy = 0
                if abs(dx) > dead_zone:
                    vy =  speed if dx > 0 else -speed   # 右/左
                if abs(dy) > dead_zone:
                    vx =  speed if dy < 0 else -speed   # 前/后
                tello.send_rc_control(vx, vy, 0, 0)

            # 右摇杆：上升/下降
            elif dist((x, y), right_js) < right_rad:
                dz = y - right_js[1]
                vz =  speed if dz < 0 else -speed       # 上/下
                tello.send_rc_control(0, 0, vz, 0)
            else:
                # 手指不在任何摇杆范围内 -> 悬停
                tello.send_rc_control(0, 0, 0, 0)

    # 6-5 显示
    cv2.imshow("Fast Tello", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------
#  7. 清理：确保线程、无人机、摄像头安全退出
# ----------------------------------------------------------------
cam_thread.stop()      # 通知读帧线程结束
cam_thread.join()      # 等待线程真正退出
tello.send_rc_control(0, 0, 0, 0)
tello.land()
tello.disconnect()
cap.release()
cv2.destroyAllWindows()
