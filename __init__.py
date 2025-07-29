"""
DJ-TelloPy 无人机控制项目

该项目提供了多种控制Tello无人机的方式，包括键盘控制、手势控制、语音控制、手掌追踪和人脸识别追踪等功能。
"""

__version__ = "1.0.0"
__author__ = "RUILii"

# 从子模块中导出常用类和函数，简化导入
from .KeyboardControlTello import FrontEnd as KeyboardController
from .PalmControlTello import tello as palm_control_tello
from .PalmTracking import tello as palm_tracking_tello
from .FaceTracking import FrontEnd as FaceTracker
from .GestureControlTello import tello as gesture_control_tello

# 包的公共接口定义
__all__ = [
    "KeyboardController",
    "palm_control_tello",
    "palm_tracking_tello",
    "FaceTracker",
    "gesture_control_tello",
    "__version__",
    "__author__"
]


# 包初始化时的信息提示
def _init_package():
    print(f"DJ-TelloPy 版本 {__version__} 加载成功")
    print("支持的控制模式：键盘控制、手势控制、手掌追踪、人脸追踪")


# 执行初始化
_init_package()
