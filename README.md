# DJ-TelloPy 无人机控制项目

一个基于 Python 的 Tello 无人机多模式控制项目，支持键盘、手势、手掌追踪、人脸识别和语音控制等多种交互方式。

## 📋 项目介绍

本项目提供了多种控制 Tello 无人机的解决方案，旨在通过不同的交互方式实现对无人机的精准控制，适合无人机爱好者和开发者学习与扩展。

### 🚀 主要功能

- **键盘控制**：通过方向键和字母键控制无人机飞行
- **手势控制**：通过特定手势指令控制无人机
- **手掌追踪**：追踪手掌位置实现无人机跟随
- **人脸识别追踪**：自动识别并追踪人脸
- **语音控制**：通过中文语音指令控制无人机
- **WiFi配置**：帮助Tello连接到WiFi网络
- **电池监控**：实时查看无人机电池状态
- **模拟器支持**：提供Tello模拟器用于测试

## 🛠️ 环境准备

### 依赖安装

项目依赖以下 Python 库：

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install djitellopy~=2.5.0 opencv-python~=4.6.0.66 pygame~=2.6.1 mediapipe~=0.10.21 PyAudio~=0.2.14 vosk~=0.3.45 numpy
```

### 硬件要求

- **Tello 无人机**（Tello EDU 或 Tello）
- **计算机**（支持 Python 3.7+）
- **摄像头**（内置或外置，用于视觉识别）
- **麦克风**（用于语音控制）

### 模型文件

项目使用以下AI模型：
- **人脸识别**：DNN模型（deploy.prototxt + res10_300x300_ssd_iter_140000_fp16.caffemodel）
- **语音识别**：Vosk中文模型（vosk-model-small-cn-0.22）

## 📖 使用方法

### 1. 键盘控制

```bash
python KeyboardControlTello.py
```

**操作说明：**
- `T`：起飞
- `L`：降落
- `方向键`：前后左右移动
- `A/D`：逆时针/顺时针旋转
- `W/S`：上升/下降
- `ESC`：退出程序

### 2. 手势控制

```bash
python GestureControlTello.py
```

**支持的手势：**
- **拳头**：起飞/降落
- **手掌**：悬停
- **OK手势**：拍照
- **比心手势**：翻转
- **其他手势**：移动控制

### 3. 手掌追踪

```bash
python PalmTracking.py
```

**功能特点：**
- 实时追踪手掌位置
- 根据手掌移动控制无人机
- 支持距离调节（手掌大小控制）
- 自动悬停功能

### 4. 人脸识别追踪

```bash
python FaceTracking.py
```

**功能特点：**
- 自动检测和追踪人脸
- 支持DNN和Haar级联两种检测方式
- 可切换自动追踪和手动控制模式
- 实时视频流显示

### 5. 语音控制

```bash
python SpeechRecognition.py
```

**支持的语音指令：**
- "起飞"：无人机起飞
- "降落"：无人机降落
- "前进"：向前移动
- "后退"：向后移动
- "左转"：向左旋转
- "右转"：向右旋转
- "上"：向上移动
- "下"：向下移动
- "结束"：退出程序

### 6. WiFi配置

```bash
python ConnectHotspot.py
```

**使用步骤：**
1. 修改代码中的WiFi名称和密码
2. 运行脚本连接Tello热点
3. 程序会自动配置Tello连接到指定WiFi

### 7. 电池监控

```bash
python GetBattery.py
```

显示当前无人机电池电量。

### 8. 模拟器测试

```bash
# 启动模拟器
python Tello_simulator.py

# 在另一个终端测试
python wifi.py
```

## 📁 项目结构

```
drone/
├── __init__.py                 # 包初始化文件
├── KeyboardControlTello.py    # 键盘控制模块
├── GestureControlTello.py     # 手势控制模块
├── PalmTracking.py            # 手掌追踪模块
├── PalmControlTello.py        # 手掌控制模块
├── FaceTracking.py            # 人脸识别追踪模块
├── SpeechRecognition.py       # 语音控制模块
├── ConnectHotspot.py          # WiFi配置模块
├── GetBattery.py              # 电池监控模块
├── Tello_simulator.py         # 模拟器模块
├── wifi.py                    # 模拟器测试客户端
├── requirements.txt           # 依赖包列表
├── model/                     # AI模型文件
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│   ├── haarcascade_frontalface_default.xml
│   └── vosk-model-small-cn-0.22/
└── photo/                     # 照片保存目录
```

## ⚠️ 注意事项

1. **安全第一**：请在开阔、安全的环境中使用无人机
2. **电池检查**：飞行前确保电池电量充足（建议>20%）
3. **网络连接**：确保设备与Tello在同一WiFi网络或热点下
4. **摄像头权限**：首次使用需要授权摄像头和麦克风权限
5. **模型文件**：确保AI模型文件完整，否则会降级使用备用方案

## 🔧 故障排除

### 常见问题

1. **连接失败**
   - 检查WiFi连接
   - 确认Tello已开机
   - 重启Tello和程序

2. **摄像头无法打开**
   - 检查摄像头权限
   - 确认摄像头未被其他程序占用

3. **语音识别不准确**
   - 确保环境安静
   - 检查麦克风权限
   - 尝试重新下载语音模型

4. **手势识别失败**
   - 确保光线充足
   - 保持手部在摄像头范围内
   - 调整检测置信度参数

## 📝 版本信息

- **版本**：1.0.0
- **作者**：RUILii
- **Python版本**：3.7+
- **最后更新**：2025年

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。





