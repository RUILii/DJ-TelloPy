import numpy as np
from djitellopy import Tello
import time


# 获取无人机电池电量
tello = Tello()
tello.connect()
print("电池电量：", tello.get_battery(), "%")

