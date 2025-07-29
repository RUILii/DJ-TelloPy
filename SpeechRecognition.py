from vosk import Model, KaldiRecognizer
import pyaudio
from djitellopy import Tello
import json
import time

# ---------- 可调参数 ----------
DISTANCE_CM = 50
SPEED_CM_S = 20
FLIGHT_TIME = DISTANCE_CM / SPEED_CM_S

# ---------- 初始化 ----------
tello = Tello()
tello.connect()
tello.send_rc_control(0, 0, 0, 0)

model = Model("model/vosk-model-small-cn-0.22")
rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=1500)

bat = tello.get_battery()
print("电池电量：", bat, "%")
if bat < 20:
    print("电量 < 20 %，拒绝起飞")
    exit()

print("离线中文语音：起飞 / 降落 / 前进 / 后退 / 左转 / 右转 / 上 / 下 / 结束")


def safe_takeoff():
    """起飞 + 异常重试"""
    for _ in range(3):
        try:
            tello.takeoff()
            time.sleep(2)      # 等待电机解锁
            tello.send_rc_control(0, 0, 0, 0)
            print("已起飞")
            return
        except Exception as e:
            print("起飞失败:", e)
            time.sleep(1)
    print("起飞失败，退出")
    exit()


# ---------- 指令映射 ----------
cmds = {
    "起飞": lambda: safe_takeoff(),
    "降落": lambda: tello.land(),
    "前进": lambda: one_shot(0,  SPEED_CM_S,  0,  0),
    "后退": lambda: one_shot(0, -SPEED_CM_S,  0,  0),
    "左转": lambda: one_shot(-SPEED_CM_S,  0,  0,  0),
    "右转": lambda: one_shot(SPEED_CM_S,  0,  0,  0),
    "上升": lambda: one_shot(0,  0,  SPEED_CM_S, 0),
    "下降": lambda: one_shot(0,  0, -SPEED_CM_S, 0),
    "结束": lambda: stop_and_exit(),
}


def one_shot(left, forward, up, y):
    if forward > 0:
        dir_name = "前进"
    elif forward < 0:
        dir_name = "后退"
    elif left > 0:
        dir_name = "右转"
    elif left < 0:
        dir_name = "左转"
    elif up > 0:
        dir_name = "上升"  # u>0 应该对应上升
    elif up < 0:
        dir_name = "下降"  # u<0 应该对应下降
    else:
        dir_name = "未知"

    print(f"开始{dir_name} {DISTANCE_CM} cm …")
    tello.send_rc_control(left, forward, up, y)
    time.sleep(FLIGHT_TIME)
    tello.send_rc_control(0, 0, 0, 0)
    print("保持悬停")


def stop_and_exit():
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    print("程序结束")
    stream.stop_stream()
    stream.close()
    p.terminate()
    exit()


# ---------- 主循环 ----------
current_command = None  # 跟踪当前执行的命令
command_start_time = 0  # 命令开始时间

while True:
    try:
        data = stream.read(200, exception_on_overflow=False)
    except OSError:
        print("麦克风异常，重新初始化...")
        stream.stop_stream()
        stream.close()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=1500)
        continue

    # 如果有命令正在执行，跳过新命令处理
    if current_command and (time.time() - command_start_time) < FLIGHT_TIME + 0.5:
        continue

    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())["text"]

        for cmd, action in cmds.items():
            if cmd in res:
                print(f"识别：{cmd}")
                current_command = cmd  # 标记当前执行的命令
                command_start_time = time.time()  # 记录开始时间

                # 清除识别器状态防止重复识别
                rec = KaldiRecognizer(model, 16000)

                action()

                # 起飞后从字典删除
                if cmd == "起飞":
                    del cmds["起飞"]

                # 重置当前命令
                current_command = None
                break

    # ---------- 结束 ----------
