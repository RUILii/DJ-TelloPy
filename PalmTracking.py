import cv2
import mediapipe as mp
from djitellopy import Tello
import time

# --- 可调参数 ---
# 您可以在这里调整无人机的行为，而无需修改主代码
CONTROL_SPEED_LR = 40  # 左右移动的最大速度 (10-100)
CONTROL_SPEED_UD = 50  # 上下移动的最大速度 (10-100)
CONTROL_SPEED_FB = 30  # 前后移动的最大速度 (10-100)
HOVER_THRESHOLD = 60   # 中心悬停区域的容忍范围（像素），手掌移动超过这个距离无人机才开始响应
DESIRED_HAND_SIZE = 150  # 理想的手掌在画面中的宽度（像素），用于控制无人机与手的距离
FB_THRESHOLD = 20      # 前后移动的宽度容忍范围（像素）

MAX_HANDS = 2          # 核心要求修改：设置最多检测2只手
MIN_CONFIDENCE = 0.7   # 手部检测的最低置信度

# --- 初始化 ---
# 1. 初始化 Tello
tello = Tello()

# 2. 初始化 MediaPipe
mpHands = mp.solutions.hands
# 根据上面的参数初始化Hands模块
hands = mpHands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=MIN_CONFIDENCE)
mpDraw = mp.solutions.drawing_utils

# --- 主程序 ---
# 使用 try...finally 结构确保无论发生什么，无人机总能安全降落
try:
    # 3. 连接、检查与起飞
    print("正在连接 Tello无人机...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Tello 电量: {battery}%")
    if battery < 20:
        raise RuntimeError("电池电量过低，为了安全，飞行已取消！")

    tello.streamon()
    frame_read = tello.get_frame_read()  # 使用后台读取器，效率更高

    print("发送起飞指令...")
    tello.takeoff()
    tello.move_up(40)  # 起飞后向上飞一点，为手掌留出空间，也更安全
    time.sleep(1)

    # 4. 主循环：实时追踪与控制
    while True:
        # 从后台读取器获取最新一帧图像
        frame = frame_read.frame
        # 健壮性检查：如果视频流中断，则跳过本次循环
        if frame is None:
            continue

        # 图像预处理
        img = cv2.resize(frame, (640, 480))
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用MediaPipe进行手部检测
        results = hands.process(imgRGB)

        # 安全机制：在每次循环开始时，都将所有速度指令重置为0
        # 这是确保“当手从画面中消失后，无人机立刻悬停”的关键
        lr, fb, ud = 0, 0, 0

        # 如果检测到了手
        if results.multi_hand_landmarks:
            # 注意：即使检测到多只手，这里的逻辑也只处理检测到的第一只手 [0]
            # 这是为了避免两只手发出矛盾的指令，导致无人机行为混乱
            handLms = results.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # --- 计算手掌的边界框、中心点和大小 ---
            x_coords = [lm.x * w for lm in handLms.landmark]
            y_coords = [lm.y * h for lm in handLms.landmark]
            hand_xmin, hand_xmax = min(x_coords), max(x_coords)
            hand_ymin, hand_ymax = min(y_coords), max(y_coords)
            
            hand_width = hand_xmax - hand_xmin
            
            palm_cx = int((hand_xmin + hand_xmax) / 2)
            palm_cy = int((hand_ymin + hand_ymax) / 2)

            # 在画面上绘制矩形框和中心点，方便调试
            cv2.rectangle(img, (int(hand_xmin), int(hand_ymin)), (int(hand_xmax), int(hand_ymax)), (0, 255, 0), 2)
            cv2.circle(img, (palm_cx, palm_cy), 10, (0, 0, 255), cv2.FILLED)

            # --- 剪刀手(V手势)检测与拍照 ---
            # 剪刀手/比V手势：食指和中指伸直，其余手指弯曲
            # MediaPipe关键点: 8(食指尖), 12(中指尖), 6(食指根), 10(中指根), 16(无名指尖), 20(小指尖)
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append((id, int(lm.x * w), int(lm.y * h)))

            is_v_sign = False
            if len(lmList) == 21:
                # 食指和中指竖直，且无名指和小指弯曲
                finger8 = lmList[8][2] < lmList[6][2]     # 食指
                finger12 = lmList[12][2] < lmList[10][2]  # 中指
                finger16 = lmList[16][2] > lmList[14][2]  # 无名指
                finger20 = lmList[20][2] > lmList[18][2]  # 小指
                if finger8 and finger12 and finger16 and finger20:
                    is_v_sign = True

            # 拍照逻辑
            if is_v_sign:
                photo_name = f"photo_{int(time.time())}.jpg"
                cv2.imwrite(photo_name, img)
                print(f"检测到剪刀手，已自动拍照并保存为 {photo_name}")
                time.sleep(1.5)  # 拍照后延迟，防止连续多次拍照


            # --- 应用平滑的“比例控制”算法 ---
            frame_cx, frame_cy = w // 2, h // 2
            offset_x = palm_cx - frame_cx
            offset_y = palm_cy - frame_cy

            # 计算左右速度（正为右，负为左）
            if abs(offset_x) > HOVER_THRESHOLD:
                lr = int(CONTROL_SPEED_LR * offset_x / frame_cx)
            
            # 计算上下速度（正为上，负为下）
            if abs(offset_y) > HOVER_THRESHOLD:
                ud = int(-1 * CONTROL_SPEED_UD * offset_y / frame_cy)  # 图像Y轴向下为正，需反转

            # 计算前后速度（正为前，负为后）
            error_fb = DESIRED_HAND_SIZE - hand_width
            if abs(error_fb) > FB_THRESHOLD:
                fb = int(CONTROL_SPEED_FB * error_fb / DESIRED_HAND_SIZE)

            # --- 应用速度限制（安全钳），防止无人机移动过快 ---
            lr = max(min(lr, CONTROL_SPEED_LR), -CONTROL_SPEED_LR)
            ud = max(min(ud, CONTROL_SPEED_UD), -CONTROL_SPEED_UD)
            fb = max(min(fb, CONTROL_SPEED_FB), -CONTROL_SPEED_FB)

        # 无论是否检测到手，都将计算出的指令（或0）发送给无人机
        tello.send_rc_control(lr, fb, ud, 0)

        # 显示追踪画面
        cv2.imshow("Tello Palm Tracking", img)
        
        # 按 'q' 键退出主循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("收到退出指令...")
            break

except Exception as e:
    # 捕获所有可能发生的异常（如连接失败、电量低等），并打印错误信息
    print(f"\n程序发生严重错误: {e}\n")

finally:
    # 5. 安全清理程序
    # 无论程序是正常退出还是异常崩溃，这个代码块都保证会被执行
    print("执行安全清理程序...")
    try:
        # 在降落前，先发送一次悬停指令，作为双重保险
        tello.send_rc_control(0, 0, 0, 0)
        # 检查无人机是否仍在飞行，如果是，则执行降落
        if tello.is_flying:
            print("正在降落无人机...")
            tello.land()
    except Exception as land_err:
        # 如果降落过程中也发生错误，打印出来，但不影响后续清理
        print(f"尝试降落时发生错误: {land_err}")
    
    # 尝试关闭视频流和所有OpenCV窗口
    try:
        tello.streamoff()
        cv2.destroyAllWindows()
    except Exception as cleanup_err:
        print(f"清理视频资源时发生错误: {cleanup_err}")

    # 最终确保Tello连接被彻底关闭
    tello.end()
    print("程序已安全退出。")
