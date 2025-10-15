from djitellopy import Tello

# 创建 Tello 对象
tello = Tello()

print("尝试连接 Tello 无人机的热点...")
try:
    # 连接到 Tello 的 Wi-Fi 热点。
    # djitellopy 库在 connect() 内部会自动发送 'command' 命令。
    tello.connect()
    print("成功连接到 Tello。")

    # --- 在这里添加你的 Wi-Fi 网络名称和密码 ---
    # 请将 'YOUR_WIFI_SSID' 替换为你的 Wi-Fi 名称
    # 请将 'YOUR_WIFI_PASSWORD' 替换为你的 Wi-Fi 密码
    wifi_ssid = "YOUR_WIFI_SSID"
    wifi_password = "YOUR_WIFI_PASSWORD"
    
    # 构建 ap 命令
    ap_command = f"ap {wifi_ssid} {wifi_password}"
    
    print(f"正在发送配置命令: '{ap_command}'")
    
    # 发送 ap 命令并接收 Tello 的响应
    # 注意：Tello 收到此命令后会断开连接，因此可能不会返回 'ok'
    response = tello.send_command_with_return(ap_command)
    
    print(f"收到 Tello 的响应: {response}")

    print("\nTello 正在尝试连接到你的 Wi-Fi 网络...")
    print("你的电脑与 Tello 的连接将断开。请手动连接到你的 Wi-Fi 网络。")
    print("配置完成后，Tello 的新 IP 地址将由你的路由器分配。")
    
except Exception as e:
    print(f"连接或命令发送失败: {e}")

finally:
    # 关闭连接
    tello.end()
    print("程序结束")
