import socket
import threading
import time
import math
import random
import logging

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


class SimpleDroneSimulator:
    def __init__(self, host='0.0.0.0', command_port=8889, state_port=9000):
        self.host = host
        self.command_port = command_port
        self.state_port = state_port
        self.client_ip = None
        self.running = False

        # 模拟无人机初始状态
        self.state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
            'bat': 100,
            'height': 0,
            'temp': 25,
            'is_flying': False,
            'motors_on': False,
        }

        self.command_socket = None
        self.state_socket = None
        self.state_thread = None

        logging.info(f"Drone simulator initialized. Will listen on {host}:{command_port}")

    def start(self):
        """启动模拟器"""
        try:
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.command_socket.bind((self.host, self.command_port))
            logging.info(f"Command socket bound to {self.host}:{self.command_port}")

            self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logging.info("State socket created")

            self.running = True
            self.state_thread = threading.Thread(target=self._send_state_loop, daemon=True)
            self.state_thread.start()
            logging.info("State thread started")

            self._receive_commands()

        except Exception as e:
            logging.error(f"Failed to start simulator: {e}")
            self.stop()

    def _receive_commands(self):
        """接收和处理UDP命令"""
        logging.info("Waiting for commands...")

        while self.running:
            try:
                data, addr = self.command_socket.recvfrom(1024)
                if self.client_ip is None:  # 记录首次连接的客户端IP
                    self.client_ip = addr[0]
                    logging.info(f"Client connected from {addr}, state will be sent to port {self.state_port}")

                command = data.decode('utf-8').strip().lower()
                logging.info(f"Command received: '{command}' from {addr}")

                response = self._process_command(command)

                if response:
                    self.command_socket.sendto(response.encode('utf-8'), addr)
                    logging.info(f"Sent response: '{response}' to {addr}")

            except socket.error as e:
                if self.running:
                    logging.error(f"Socket error in command receiver: {e}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in command receiver: {e}")
                time.sleep(1)

    def _process_command(self, command):
        """处理接收到的命令并更新无人机状态"""
        response = "ok"

        if command == "command":
            self.state['motors_on'] = True
            logging.info("Entered SDK mode")

        elif command == "takeoff":
            if not self.state['is_flying']:
                self.state['is_flying'] = True
                self.state['z'] = 1.0
                logging.info("Takeoff command executed")
            else:
                response = "error: already flying"
                logging.warning("Takeoff command rejected: already flying")

        elif command == "land":
            if self.state['is_flying']:
                self.state['is_flying'] = False
                self.state['z'] = 0.0
                self.state['vx'], self.state['vy'], self.state['vz'] = 0.0, 0.0, 0.0
                logging.info("Land command executed")
            else:
                response = "error: not flying"
                logging.warning("Land command rejected: not flying")

        elif command.startswith("forward"):
            try:
                dist = int(command.split()[1])
                self.state['y'] += dist / 100.0
                logging.info(f"Forward command: {dist} cm")
            except (IndexError, ValueError):
                response = "error: invalid parameter"

        elif command.startswith("back"):
            try:
                dist = int(command.split()[1])
                self.state['y'] -= dist / 100.0
                logging.info(f"Back command: {dist} cm")
            except (IndexError, ValueError):
                response = "error: invalid parameter"

        elif command.startswith("left"):
            try:
                dist = int(command.split()[1])
                self.state['x'] -= dist / 100.0
                logging.info(f"Left command: {dist} cm")
            except (IndexError, ValueError):
                response = "error: invalid parameter"

        elif command.startswith("right"):
            try:
                dist = int(command.split()[1])
                self.state['x'] += dist / 100.0
                logging.info(f"Right command: {dist} cm")
            except (IndexError, ValueError):
                response = "error: invalid parameter"

        elif command.startswith("cw"):
            try:
                angle = int(command.split()[1])
                self.state['yaw'] = (self.state['yaw'] + angle) % 360
                logging.info(f"Clockwise rotation: {angle} degrees")
            except (IndexError, ValueError):
                response = "error: invalid parameter"

        elif command.startswith("ccw"):
            try:
                angle = int(command.split()[1])
                self.state['yaw'] = (self.state['yaw'] - angle) % 360
                logging.info(f"Counter-clockwise rotation: {angle} degrees")
            except (IndexError, ValueError):
                response = "error: invalid parameter"

        elif command.startswith("flip"):
            # 检查是否处于飞行状态
            if not self.state['is_flying']:
                response = "error: not flying"
                logging.warning("Flip command rejected: not flying")
            else:
                try:
                    # 解析翻转方向
                    direction = command.split()[1]
                    if direction == 'f':
                        logging.info("Simulating: Executing flip forward")
                    elif direction == 'b':
                        logging.info("Simulating: Executing flip backward")
                    elif direction == 'l':
                        logging.info("Simulating: Executing flip left")
                    elif direction == 'r':
                        logging.info("Simulating: Executing flip right")
                    else:
                        response = "error: invalid flip direction"
                        logging.warning(f"Invalid flip direction: {direction}")
                except IndexError:
                    response = "error: missing flip direction"
                    logging.warning("Flip command missing direction parameter")

        elif command == "battery?":
            response = str(self.state['bat'])

        elif command == "speed?":
            speed = math.sqrt(self.state['vx'] ** 2 + self.state['vy'] ** 2 + self.state['vz'] ** 2)
            response = str(int(speed * 100))

        elif command == "temp?":
            response = str(self.state['temp'])

        elif command == "time?":
            response = "15"  # 模拟飞行时间

        else:
            if command != "command":  # 'command' 指令不应显示为未知
                response = "error: unknown command"
                logging.warning(f"Unknown command: {command}")

        # 模拟电量消耗
        if self.state['is_flying'] and random.random() < 0.3:
            self.state['bat'] = max(0, self.state['bat'] - 1)
            if self.state['bat'] % 10 == 0:  # 每消耗10%电量时提示一次
                logging.info(f"Battery decreased to {self.state['bat']}%")

        return response

    def _send_state_loop(self):
        """在独立线程中循环发送状态信息"""
        logging.info("State sender thread started")
        while self.running:
            try:
                if not self.client_ip:
                    time.sleep(0.5)
                    continue

                target_addr = (self.client_ip, self.state_port)

                if self.state['is_flying']:
                    self.state['vx'] += (random.random() - 0.5) * 0.1
                    self.state['vy'] += (random.random() - 0.5) * 0.1
                    self.state['temp'] = 25 + random.randint(0, 2)

                state_str = (
                    f"pitch:{self.state['pitch']:.1f};roll:{self.state['roll']:.1f};yaw:{self.state['yaw']:.1f};"
                    f"vx:{self.state['vx']:.2f};vy:{self.state['vy']:.2f};vz:{self.state['vz']:.2f};"
                    f"bat:{self.state['bat']};temp:{self.state['temp']};height:{int(self.state['z'] * 100)}")

                self.state_socket.sendto(state_str.encode('utf-8'), target_addr)
                logging.debug(f"State sent to {target_addr}")

                time.sleep(0.2)

            except Exception as e:
                logging.error(f"Error in state sender: {e}")
                time.sleep(1)

    def stop(self):
        """停止模拟器"""
        self.running = False
        if self.command_socket:
            self.command_socket.close()
        if self.state_socket:
            self.state_socket.close()
        logging.info("Drone simulator stopped")


# 主程序入口
if __name__ == "__main__":
    # 确保 host 是 '0.0.0.0' 以接收来自模拟器的连接
    simulator = SimpleDroneSimulator(host='0.0.0.0', command_port=8889, state_port=9000)
    try:
        simulator.start()
    except KeyboardInterrupt:
        print("\nShutting down simulator...")
    finally:
        simulator.stop()
