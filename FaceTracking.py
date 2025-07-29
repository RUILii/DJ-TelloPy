from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import os

MODEL_DIR = "D:/drone_models/"

S = 30  # 键盘控制速度
FPS = 120  # 帧率
TRACK_SPEED = 25  # 跟踪移动速度
DEAD_ZONE = 60  # 中心死区范围(不响应微小移动)


class FrontEnd(object):
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])
        self.tello = Tello()

        # 速度控制变量
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.send_rc_control = False
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        # ====== 人脸识别模型加载 ======
        self.use_dnn = False
        self.face_cascade = None
        self.net = None
        prototxt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
        model_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                if not self.net.empty():
                    self.use_dnn = True
                    print("✅ DNN 模型加载成功")
            except Exception as e:
                print("❌ DNN 加载失败:", e)
        if not self.use_dnn:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("⚠️  降级到 Haar 级联")

        # 控制模式
        self.control_mode = "AUTO"  # AUTO: 自动跟踪模式, MANUAL: 键盘控制模式
        self.last_mode_change_time = 0

        # 键盘控制状态跟踪
        self.keyboard_control_active = False
        self.last_keyboard_time = 0
        self.keyboard_timeout = 1.0  # 键盘控制超时时间（秒）

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)
        self.tello.streamoff()
        self.tello.streamon()
        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])
            frame = frame_read.frame
            frame = cv2.flip(frame, 1)  # 镜像

            # ====== 人脸检测 ======
            face_found = False
            largest_face = None

            # DNN检测逻辑
            if self.use_dnn:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > 0.7:  # 提高置信度阈值
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        face_area = (x2 - x1) * (y2 - y1)

                        # 保存最大的人脸
                        if largest_face is None or face_area > largest_face[4]:
                            largest_face = (x1, y1, x2, y2, face_area, conf)

            # Haar级联检测逻辑
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)  # 调整参数提高准确性

                for (x, y, w, h) in faces:
                    face_area = w * h
                    # 保存最大的人脸
                    if largest_face is None or face_area > largest_face[4]:
                        largest_face = (x, y, x + w, y + h, face_area, 1.0)

            # ====== 跟踪逻辑 ======
            FRAME_CX, FRAME_CY = 480, 360  # 画面中心点(960x720的中心)

            # 检查键盘控制是否超时
            if time.time() - self.last_keyboard_time > self.keyboard_timeout:
                self.keyboard_control_active = False

            # 自动跟踪模式
            if self.control_mode == "AUTO":
                if largest_face and not self.keyboard_control_active:
                    # 获取人脸中心点坐标
                    if self.use_dnn:
                        x1, y1, x2, y2, area, conf = largest_face
                    else:
                        x1, y1, x2, y2, area, _ = largest_face

                    face_cx = (x1 + x2) // 2
                    face_cy = (y1 + y2) // 2

                    # 计算与画面中心的偏差
                    dx = face_cx - FRAME_CX
                    dy = face_cy - FRAME_CY

                    # 绘制跟踪信息
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
                    cv2.line(frame, (FRAME_CX, FRAME_CY), (face_cx, face_cy), (255, 0, 0), 2)

                    # 仅在起飞状态下跟踪
                    if self.send_rc_control:
                        # 水平方向跟踪 (左右平移)
                        if abs(dx) > DEAD_ZONE:
                            # 修正：方向调整（当人脸在右侧时，无人机应向左移动）
                            # 因为无人机移动方向与画面方向相反
                            self.left_right_velocity = TRACK_SPEED if dx < 0 else -TRACK_SPEED
                        else:
                            self.left_right_velocity = 0

                        # 垂直方向跟踪 (升高/降低)
                        if abs(dy) > DEAD_ZONE:
                            # 修正：方向调整（当人脸在下方时，无人机应上升）
                            # 注意：Tello坐标系中，正值为上升，负值为下降
                            self.up_down_velocity = TRACK_SPEED if dy < 0 else -TRACK_SPEED
                        else:
                            self.up_down_velocity = 0

                        # 始终0速度控制 (保持高度)
                        self.for_back_velocity = 0
                        self.yaw_velocity = 0
                else:
                    # 未检测到人脸时停止移动 (悬停)
                    if self.send_rc_control and not self.keyboard_control_active:
                        self.left_right_velocity = 0
                        self.up_down_velocity = 0
                        self.for_back_velocity = 0
                        self.yaw_velocity = 0

                    # 绘制中心标记
                    cv2.circle(frame, (FRAME_CX, FRAME_CY), 10, (0, 0, 255), -1)

            # 手动控制模式
            elif self.control_mode == "MANUAL":
                # 在手动模式下，不执行自动跟踪
                if self.send_rc_control:
                    # 如果用户没有操作键盘，停止移动
                    if not self.keyboard_control_active:
                        self.left_right_velocity = 0
                        self.up_down_velocity = 0
                        self.for_back_velocity = 0
                        self.yaw_velocity = 0

                # 绘制中心标记
                cv2.circle(frame, (FRAME_CX, FRAME_CY), 10, (0, 0, 255), -1)

            # 显示控制模式
            mode_text = f"MODE: {self.control_mode}"
            cv2.putText(frame, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示跟踪状态
            if self.send_rc_control:
                if self.control_mode == "AUTO":
                    if largest_face and not self.keyboard_control_active:
                        status_text = "TRACKING"
                        color = (0, 255, 0)
                    elif self.keyboard_control_active:
                        status_text = "KEYBOARD CONTROL"
                        color = (255, 255, 0)
                    else:
                        status_text = "HOVERING"
                        color = (0, 0, 255)
                else:  # MANUAL模式
                    if self.keyboard_control_active:
                        status_text = "MANUAL CONTROL"
                        color = (255, 255, 0)
                    else:
                        status_text = "HOVERING"
                        color = (0, 0, 255)

                cv2.putText(frame, status_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 显示人脸位置偏差信息
            if largest_face and self.control_mode == "AUTO":
                if self.use_dnn:
                    x1, y1, x2, y2, area, conf = largest_face
                else:
                    x1, y1, x2, y2, area, _ = largest_face

                face_cx = (x1 + x2) // 2
                face_cy = (y1 + y2) // 2
                dx = face_cx - FRAME_CX
                dy = face_cy - FRAME_CY

                # 显示偏差信息
                pos_text = f"DX: {dx}, DY: {dy}"
                cv2.putText(frame, pos_text, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 显示速度信息
                vel_text = f"LR: {self.left_right_velocity}, UD: {self.up_down_velocity}"
                cv2.putText(frame, vel_text, (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()
            time.sleep(1 / FPS)

        self.tello.end()

    def keydown(self, key):
        # 标记键盘控制活动状态
        self.keyboard_control_active = True
        self.last_keyboard_time = time.time()

        if key == pygame.K_UP:  # 前进
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # 后退
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # 左方向键
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # 右方向键
            self.left_right_velocity = S
        elif key == pygame.K_w:  # w 上升
            self.up_down_velocity = S
        elif key == pygame.K_s:  # s 下降
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # a 左转
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # d 右转
            self.yaw_velocity = S
        elif key == pygame.K_t:  # t 起飞
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # l 降落
            self.tello.land()
            self.send_rc_control = False
            self.keyboard_control_active = False
        elif key == pygame.K_SPACE:  # 空格键切换控制模式
            if self.control_mode == "AUTO":
                self.control_mode = "MANUAL"
                print("切换到手动控制模式")
            else:
                self.control_mode = "AUTO"
                print("切换到自动跟踪模式")
            # 重置键盘控制状态
            self.keyboard_control_active = False

    def keyup(self, key):
        # 标记键盘控制活动状态
        self.keyboard_control_active = True
        self.last_keyboard_time = time.time()

        if key in (pygame.K_UP, pygame.K_DOWN):
            self.for_back_velocity = 0
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            self.left_right_velocity = 0
        elif key in (pygame.K_w, pygame.K_s):
            self.up_down_velocity = 0
        elif key in (pygame.K_a, pygame.K_d):
            self.yaw_velocity = 0

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity,
                                       self.for_back_velocity,
                                       self.up_down_velocity,
                                       self.yaw_velocity)


def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()
