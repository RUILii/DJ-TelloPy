#!/usr/bin/env python3
"""
Tello 模拟器完整测试客户端
用法:
    python test_drone.py                # 交互模式
    python test_drone.py command takeoff forward 100 land
"""
import socket
import threading
import time
import sys
import argparse
from datetime import datetime

import select

CMD_ADDR = ("127.0.0.1", 8889)
STATE_ADDR = ("0.0.0.0", 9000)   # 本机监听模拟器发来的状态
TIMEOUT = 3.0


def log(msg):
    print(f"[{datetime.now():%H:%M:%S.%f}] {msg}")


class DroneTester:
    def __init__(self):
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.settimeout(TIMEOUT)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(STATE_ADDR)
        self.state_sock.settimeout(1.0)

        self.running = True
        self.latest_state = ""          # 最新一帧
        self._last_print = 0            # 上次打印时间

        # 守护线程只负责收数据
        threading.Thread(target=self._listen_state, daemon=True).start()

    # ---------- 状态线程：只收不打印 ----------
    def _listen_state(self):
        while self.running:
            try:
                data, _ = self.state_sock.recvfrom(1024)
                self.latest_state = data.decode().strip()
            except socket.timeout:
                continue
            except Exception:
                pass

    # ---------- 定时把最新一帧打到同一行 ----------
    def _refresh_status_line(self):
        now = time.time()
        if now - self._last_print >= 2.0 and self.latest_state:
            self._last_print = now
            # \r 回到行首，\x1b[K 清空本行，末尾不换行
            sys.stdout.write("\r\x1b[K[STATE] " + self.latest_state)
            sys.stdout.flush()

    # ---------- 发送命令（不变） ----------
    def send(self, cmd: str, retry: int = 1) -> str:
        cmd = cmd.strip()
        for attempt in range(1, retry + 1):
            try:
                self.cmd_sock.sendto(cmd.encode(), CMD_ADDR)
                log(f"CMD >>> {cmd}")
                data, addr = self.cmd_sock.recvfrom(1024)
                resp = data.decode().strip()
                log(f"CMD <<< {resp}")
                return resp
            except socket.timeout:
                log(f"CMD ??? timeout (attempt {attempt}/{retry})")
                if attempt == retry:
                    return "timeout"
            except Exception as e:
                log(f"CMD !!! error: {e}")
                return str(e)
        return "unknown"

    # ---------- 交互模式：边读边刷状态 ----------
    def interactive(self):
        log("Interactive mode. Type 'quit' to exit.")
        try:
            while True:
                self._refresh_status_line()          # 2 s 刷一次
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() in {"quit", "exit"}:
                        break
                    if line:
                        self.send(line, retry=2)
                time.sleep(0.05)
        except (KeyboardInterrupt, EOFError):
            pass
        # 退出前换行，避免覆盖提示符
        print()

    def close(self):
        self.running = False
        self.cmd_sock.close()
        self.state_sock.close()


# -------------------- 两种使用模式 --------------------
def interactive_mode(tester: DroneTester):
    log("Interactive mode. Type 'quit' to exit.")
    try:
        while True:
            cmd = input(">>> ").strip()
            if cmd.lower() in {"quit", "exit"}:
                break
            if cmd:
                tester.send(cmd, retry=2)
    except (KeyboardInterrupt, EOFError):
        pass


def batch_mode(tester: DroneTester, commands):
    for cmd in commands:
        tester.send(cmd, retry=2)
        if cmd in {"takeoff", "land"}:
            # 给模拟器一点时间更新状态
            time.sleep(1.5)


# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="Tello 模拟器测试客户端")
    parser.add_argument("commands", nargs="*", help="要批量执行的命令（留空则进入交互模式）")
    args = parser.parse_args()

    tester = DroneTester()
    try:
        if args.commands:
            batch_mode(tester, args.commands)
        else:
            interactive_mode(tester)
    finally:
        tester.close()
        log("Tester closed.")


if __name__ == "__main__":
    main()
