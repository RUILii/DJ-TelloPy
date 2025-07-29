import cv2
import mediapipe as mp
import threading
import queue
import time
import os
from djitellopy import Tello

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# 初始化 Tello
tello = Tello()
tello.connect()
print(f"Tello电池: {tello.get_battery()}%")
tello.streamon()
frame_read = tello.get_frame_read()

# 电脑摄像头用于手势检测
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开电脑摄像头")
    exit()

print("电脑摄像头模式启动，按Q退出程序。")

command_queue = queue.Queue(maxsize=10)

is_takeoff = False
last_special_gesture = None
last_flip_gesture = None
last_capture_gesture = None

# 获取当前项目的根目录（即当前运行脚本所在的文件夹）
project_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出当前项目目录下的photo文件夹路径
save_dir = os.path.join(project_dir, "photo")

# 如果路径不存在，则创建文件夹
os.makedirs(save_dir, exist_ok=True)


def draw_control_zones(img):
    cv2.rectangle(img, (0, 0), (200, img.shape[0]), (255, 0, 0), 2)
    cv2.putText(img, "Move Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.rectangle(img, (440, 0), (640, img.shape[0]), (255, 0, 0), 2)
    cv2.putText(img, "Move Right", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.rectangle(img, (0, 0), (img.shape[1], 150), (0, 255, 255), 2)
    cv2.putText(img, "Move Up", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.rectangle(img, (0, 350), (img.shape[1], img.shape[0]), (0, 255, 255), 2)
    cv2.putText(img, "Move Down", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def flight_control_real():
    global is_takeoff
    while True:
        try:
            cmd = command_queue.get(timeout=1)
            print(f"[Flight Thread] 收到指令: {cmd}")

            if cmd == "move_left":
                tello.move_left(20)
            elif cmd == "move_right":
                tello.move_right(20)
            elif cmd == "move_up":
                tello.move_up(20)
            elif cmd == "move_down":
                tello.move_down(20)
            elif cmd == "takeoff":
                if not is_takeoff:
                    tello.takeoff()
                    is_takeoff = True
            elif cmd == "flip":
                tello.flip_forward()
            elif cmd == "land":
                tello.land()
                break
            elif cmd == "capture":
                # 拍照保存无人机摄像头画面
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.jpg"
                full_path = os.path.join(save_dir, filename)
                frame = frame_read.frame
                if frame is not None:
                    cv2.imwrite(full_path, frame)
                    print(f"[Flight Thread] 已保存无人机照片: {full_path}")
                else:
                    print("[Flight Thread] 无法获取无人机画面，拍照失败")

            command_queue.task_done()
        except queue.Empty:
            continue


flight_thread = threading.Thread(target=flight_control_real, daemon=True)
flight_thread.start()

current_lr_command = None
current_ud_command = None

try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("无法读取电脑摄像头帧，退出。")
            break

        img = cv2.resize(img, (640, 480))
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lr_cmd = None
        ud_cmd = None
        special_cmd = None

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                h, w, _ = img.shape
                lm = [(int(p.x * w), int(p.y * h)) for p in handLms.landmark]

                index_tip = lm[8]
                middle_tip = lm[12]
                ring_tip = lm[16]
                pinky_tip = lm[20]
                thumb_tip = lm[4]

                index_mcp = lm[5]
                middle_mcp = lm[9]
                ring_mcp = lm[13]
                pinky_mcp = lm[17]
                thumb_ip = lm[3]
                thumb_cmc = lm[2]

                index_extended = index_tip[1] < index_mcp[1]
                middle_extended = middle_tip[1] < middle_mcp[1]
                ring_folded = ring_tip[1] > ring_mcp[1]
                pinky_folded = pinky_tip[1] > pinky_mcp[1]
                thumb_folded = thumb_tip[0] < thumb_cmc[0]

                # 拍照手势：食指和中指伸直，其他手指卷曲
                if index_extended and middle_extended and ring_folded and pinky_folded and thumb_folded:
                    special_cmd = "capture"
                    cv2.putText(img, "✌️ Capture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 起飞手势：大拇指向上，其他手指卷曲
                elif (thumb_tip[1] < thumb_ip[1]
                      and index_tip[1] > index_mcp[1]
                      and middle_tip[1] > middle_mcp[1]
                      and ring_tip[1] > ring_mcp[1]
                      and pinky_tip[1] > pinky_mcp[1]):
                    special_cmd = "takeoff"
                    cv2.putText(img, "👍 Takeoff", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 翻转手势：小拇指伸直，其他手指卷曲
                pinky_up = pinky_tip[1] < pinky_mcp[1]
                index_folded = index_tip[1] > index_mcp[1]
                middle_folded = middle_tip[1] > middle_mcp[1]
                ring_folded = ring_tip[1] > ring_mcp[1]

                if pinky_up and index_folded and middle_folded and ring_folded:
                    special_cmd = "flip"
                    cv2.putText(img, "🖕 Flip", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                index_x, index_y = index_tip

                if index_x < 200:
                    lr_cmd = "move_left"
                    cv2.putText(img, "Move Left", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif index_x > 440:
                    lr_cmd = "move_right"
                    cv2.putText(img, "Move Right", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if index_y < 150:
                    ud_cmd = "move_up"
                    cv2.putText(img, "Move Up", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif index_y > 350:
                    ud_cmd = "move_down"
                    cv2.putText(img, "Move Down", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 去抖动：拍照手势只触发一次
        if special_cmd == "capture":
            if last_capture_gesture != "capture":
                if not command_queue.full():
                    command_queue.put_nowait("capture")
                last_capture_gesture = "capture"
        else:
            last_capture_gesture = None

        # 起飞手势只触发一次
        if not is_takeoff:
            if special_cmd == "takeoff" and special_cmd != last_special_gesture:
                if not command_queue.full():
                    command_queue.put_nowait(special_cmd)
                    last_special_gesture = special_cmd
            else:
                last_special_gesture = None
        else:
            # 翻转手势只触发一次
            if special_cmd == "flip":
                if last_flip_gesture != "flip":
                    if not command_queue.full():
                        command_queue.put_nowait("flip")
                        last_flip_gesture = "flip"
            else:
                last_flip_gesture = None

            # 其他特殊命令只触发一次
            if special_cmd and special_cmd not in ["flip", "takeoff", "capture"] \
                    and special_cmd != last_special_gesture:
                if not command_queue.full():
                    command_queue.put_nowait(special_cmd)
                    last_special_gesture = special_cmd
            elif special_cmd is None or special_cmd in ["flip", "takeoff", "capture"]:
                if special_cmd != "flip":
                    last_special_gesture = None

            # 左右移动
            if lr_cmd != current_lr_command and lr_cmd is not None:
                if not command_queue.full():
                    command_queue.put_nowait(lr_cmd)
                    current_lr_command = lr_cmd
            elif lr_cmd is None:
                current_lr_command = None

            # 上下移动
            if ud_cmd != current_ud_command and ud_cmd is not None:
                if not command_queue.full():
                    command_queue.put_nowait(ud_cmd)
                    current_ud_command = ud_cmd
            elif ud_cmd is None:
                current_ud_command = None

        draw_control_zones(img)
        cv2.imshow("手势控制电脑摄像头", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if not command_queue.full():
                command_queue.put_nowait("land")
            break

except Exception as e:
    print("异常:", e)

finally:
    print("执行安全降落...")
    try:
        tello.land()
    except Exception as e:
        print(f"降落异常: {e}")

    cap.release()
    cv2.destroyAllWindows()
    flight_thread.join(timeout=5)
    tello.end()
    print("程序结束")
