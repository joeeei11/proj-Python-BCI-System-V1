# -*- coding: utf-8 -*-
# device_control.py
#
# 外设控制面板（串口 / 蓝牙 / WiFi），最小侵入适配主程序：
# - 信号：info(str), device_feedback(str), send_result(bool, str)
# - 方法：handle_trial_result(pred:str, success:bool)
# - UI：连接参数、手动发送左/右、自动发送策略（仅成功才发 / 始终发送）
#
# 与 main.py 联动：
#   self.device_page.info.connect(self.on_info)
#   self.device_page.device_feedback.connect(self.on_device_feedback)
#   self.device_page.send_result.connect(self.on_device_send_result)

import socket
import logging
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QComboBox,
    QLineEdit, QPushButton, QCheckBox, QStatusBar, QMessageBox, QSpinBox
)

# 可选依赖：串口、蓝牙
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None

try:
    from bluetooth import BluetoothSocket, RFCOMM  # pip install pybluez
except Exception:
    BluetoothSocket, RFCOMM = None, None


APPLE_BLUE = "#007AFF"
APPLE_GRAY = "#F0F0F0"
DARK_TEXT = "#323232"


class ControlPanel(QWidget):
    # 对外信号
    info = pyqtSignal(str)
    device_feedback = pyqtSignal(str)
    send_result = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self._log = logging.getLogger("NeuroPilot.Device")

        # 连接对象
        self.mode = "Serial"           # "Serial" / "Bluetooth" / "WiFi"
        self.ser: Optional["serial.Serial"] = None
        self.bt: Optional["BluetoothSocket"] = None
        self.sock: Optional["socket.socket"] = None

        # 收包缓冲（WiFi/BT）
        self._rx_buffer = bytearray()

        # 设备忙碌状态（发送中）
        self._busy = False

        # 轮询定时器：读取外设反馈
        self.rx_timer = QTimer(self)
        self.rx_timer.setInterval(50)  # 20Hz
        self.rx_timer.timeout.connect(self._poll_feedback)

        self._build_ui()
        self._apply_styles()

    # ---------------- UI ----------------
    def _build_ui(self):
        # 连接参数
        conn_box = QGroupBox("连接")
        g = QGridLayout()

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["串口", "蓝牙", "WiFi(TCP)"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

        # 串口
        self.cmb_port = QComboBox()
        self.btn_refresh = QPushButton("刷新串口")
        self.btn_refresh.clicked.connect(self._refresh_serial_ports)
        self.ed_baud = QLineEdit("115200")

        # 蓝牙
        self.ed_bt_addr = QLineEdit("00:11:22:33:44:55")

        # WiFi
        self.ed_ip = QLineEdit("192.168.4.1")
        self.ed_port = QLineEdit("8080")

        self.btn_connect = QPushButton("连接")
        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setEnabled(False)

        r = 0
        g.addWidget(QLabel("模式"), r, 0); g.addWidget(self.cmb_mode, r, 1); r += 1

        g.addWidget(QLabel("串口号"), r, 0); g.addWidget(self.cmb_port, r, 1)
        g.addWidget(self.btn_refresh, r, 2)
        g.addWidget(QLabel("波特率"), r, 3); g.addWidget(self.ed_baud, r, 4); r += 1

        g.addWidget(QLabel("蓝牙地址"), r, 0); g.addWidget(self.ed_bt_addr, r, 1, 1, 2); r += 1

        g.addWidget(QLabel("WiFi IP"), r, 0); g.addWidget(self.ed_ip, r, 1)
        g.addWidget(QLabel("端口"), r, 2); g.addWidget(self.ed_port, r, 3); r += 1

        g.addWidget(self.btn_connect, r, 0)
        g.addWidget(self.btn_disconnect, r, 1)
        conn_box.setLayout(g)

        # 控制区
        ctl_box = QGroupBox("控制")
        h = QHBoxLayout()
        self.btn_left = QPushButton("发送：左手")
        self.btn_right = QPushButton("发送：右手")
        self.chk_auto = QCheckBox("自动发送（来自范式/分类）")
        self.chk_only_success = QCheckBox("仅预测成功才发送")
        self.chk_auto.setChecked(True)
        self.chk_only_success.setChecked(True)
        h.addWidget(self.btn_left); h.addWidget(self.btn_right)
        h.addStretch(1)
        h.addWidget(self.chk_auto); h.addWidget(self.chk_only_success)
        ctl_box.setLayout(h)

        # 状态区
        self.status = QStatusBar()
        self._set_status("未连接")

        # 布局
        root = QVBoxLayout()
        root.addWidget(conn_box)
        root.addWidget(ctl_box)
        root.addWidget(self.status)
        self.setLayout(root)

        # 事件
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_left.clicked.connect(lambda: self._send_by_label("left"))
        self.btn_right.clicked.connect(lambda: self._send_by_label("right"))

        # 初始化
        self._refresh_serial_ports()
        self._on_mode_changed()

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: #FFFFFF;
                color: {DARK_TEXT};
                font-family: "Microsoft YaHei","微软雅黑",Arial;
                font-size: 14px;
            }}
            QGroupBox {{
                border: 1px solid #E6E6E6;
                border-radius: 12px;
                padding: 10px;
                margin-top: 8px;
                background: #FAFAFA;
                font-weight: bold;
            }}
            QLineEdit, QComboBox {{
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                padding: 6px 10px;
                background: #F7F7F7;
                min-width: 140px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {APPLE_BLUE};
                background: #FFFFFF;
            }}
            QPushButton {{
                background: {APPLE_BLUE};
                color: #FFF;
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 120px;
            }}
            QPushButton:hover {{ background:#1A84FF; }}
            QPushButton:pressed {{ background:#0062CC; }}
            QPushButton:disabled {{ background:#E0E0E0; color:#9E9E9E; }}
        """)

    # ---------------- 连接/断开 ----------------
    def _on_mode_changed(self):
        idx = self.cmb_mode.currentIndex()
        self.mode = ["Serial", "Bluetooth", "WiFi"][idx]
        # 控件可见性
        is_serial = (self.mode == "Serial")
        is_bt = (self.mode == "Bluetooth")
        is_wifi = (self.mode == "WiFi")

        # 串口行可用性
        self.cmb_port.setEnabled(is_serial)
        self.btn_refresh.setEnabled(is_serial)
        self.ed_baud.setEnabled(is_serial)

        # 蓝牙行
        self.ed_bt_addr.setEnabled(is_bt)

        # WiFi 行
        self.ed_ip.setEnabled(is_wifi)
        self.ed_port.setEnabled(is_wifi)

        self.info.emit(f"通信模式切换为：{self.mode}")

    def _refresh_serial_ports(self):
        self.cmb_port.clear()
        if serial is None:
            self.cmb_port.addItem("未安装pyserial")
            return
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            self.cmb_port.addItem("无可用串口")
        else:
            self.cmb_port.addItems(ports)

    def _connect(self):
        try:
            if self.mode == "Serial":
                if serial is None:
                    raise RuntimeError("未安装 pyserial，无法使用串口")
                port = self.cmb_port.currentText().strip()
                if not port or "无可用" in port or "未安装" in port:
                    raise RuntimeError("未选择有效串口")
                baud = int(self.ed_baud.text().strip() or "115200")
                self.ser = serial.Serial(port, baudrate=baud, timeout=0.05)
                self.info.emit(f"串口已连接：{port}@{baud}")
                self._log.info("Serial connected: %s@%s", port, baud)

            elif self.mode == "Bluetooth":
                if BluetoothSocket is None:
                    raise RuntimeError("未安装 pybluez，无法使用蓝牙")
                addr = self.ed_bt_addr.text().strip()
                self.bt = BluetoothSocket(RFCOMM)
                self.bt.connect((addr, 1))
                self.bt.setblocking(False)
                self.info.emit(f"蓝牙已连接：{addr}")
                self._log.info("Bluetooth connected: %s", addr)

            else:  # WiFi
                ip = self.ed_ip.text().strip()
                port = int(self.ed_port.text().strip() or "8080")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(3.0)
                self.sock.connect((ip, port))
                self.sock.setblocking(False)  # 非阻塞收包
                self.info.emit(f"WiFi 已连接：{ip}:{port}")
                self._log.info("WiFi connected: %s:%s", ip, port)

            self._set_status("已连接")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self._set_btns_enabled(True)
            self.rx_timer.start()

        except Exception as e:
            self.info.emit(f"连接失败：{e}")
            self._log.error("Connect failed: %s", e)
            QMessageBox.critical(self, "连接失败", str(e))

    def _disconnect(self):
        self.rx_timer.stop()
        try:
            if self.ser:
                self.ser.close(); self.ser = None
            if self.bt:
                try: self.bt.close()
                except: pass
                self.bt = None
            if self.sock:
                try: self.sock.close()
                except: pass
                self.sock = None
        finally:
            self._set_status("未连接")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self._set_btns_enabled(False)
            self.info.emit("连接已断开")
            self._log.info("Disconnected")

    # ---------------- 发送/策略 ----------------
    def _send_by_label(self, label: str):
        """手动按钮：发送左/右命令"""
        if label not in ("left", "right"):
            return
        self._send_command(self._encode_command(label))

    def handle_trial_result(self, pred: str, success: bool):
        """
        来自 EEG/范式的回调（主程序已连接）
        - 若启用自动发送：
          - 若勾选“仅预测成功才发送”：仅 success=True 时发送 pred 对应的命令
          - 否则：不论是否成功都发送 pred 对应命令
        """
        if not self.chk_auto.isChecked():
            return
        if self.chk_only_success.isChecked() and not success:
            self.info.emit("自动发送：本次未成功，跳过发送")
            self._log.info("Auto send skipped (not success)")
            self.send_result.emit(False, "未成功，未发送")
            return
        self._send_command(self._encode_command(pred))

    def _encode_command(self, label: str) -> bytes:
        """
        定义简单串行协议：'L\\n' / 'R\\n'
        - 若你的 STM32 有固定协议（如帧头/校验），在这里统一封装
        """
        if label == "left":
            return b"L\n"
        else:
            return b"R\n"

    def _send_command(self, payload: bytes):
        """统一发送入口（串口/蓝牙/WiFi）。自动管理忙/按钮状态，结果通过 send_result 信号返回。"""
        if not self._is_connected():
            self.info.emit("未连接，无法发送")
            self._log.warning("Send failed: not connected")
            self.send_result.emit(False, "未连接")
            return
        if self._busy:
            self.info.emit("设备正忙，请稍后")
            self._log.info("Send skipped: busy")
            return

        # 设备进入 busy，按钮联动
        self._busy = True
        self._set_btns_enabled(False)

        ok, msg = False, ""
        try:
            if self.ser:
                self.ser.write(payload); self.ser.flush()
                ok, msg = True, f"串口发送 {payload!r}"
            elif self.bt:
                self.bt.send(payload)
                ok, msg = True, f"蓝牙发送 {payload!r}"
            elif self.sock:
                self.sock.sendall(payload)
                ok, msg = True, f"WiFi发送 {payload!r}"
            else:
                ok, msg = False, "未知连接"
        except Exception as e:
            ok, msg = False, f"发送失败：{e}"

        # 立刻反馈一次
        self.send_result.emit(ok, msg)
        self.info.emit(msg)
        if ok:
            self._log.info(msg)
        else:
            self._log.error(msg)

        # 设置一个短暂的“忙等待”后恢复（若你的设备有 ACK，可在 _poll_feedback 收到 ACK 后再解忙）
        QTimer.singleShot(250, self._release_busy)

    def _release_busy(self):
        self._busy = False
        if self._is_connected():
            self._set_btns_enabled(True)

    def _is_connected(self) -> bool:
        return any([self.ser is not None, self.bt is not None, self.sock is not None])

    def _set_btns_enabled(self, enabled: bool):
        # 只有在连接状态下才根据 busy 开关按钮
        self.btn_left.setEnabled(enabled and self._is_connected())
        self.btn_right.setEnabled(enabled and self._is_connected())

    def _set_status(self, text: str):
        self.status.showMessage(f"设备状态：{text}")

    # ---------------- 收包/反馈 ----------------
    def _poll_feedback(self):
        """轮询读取设备返回，统一按行分帧（\\n 切分）"""
        try:
            if self.ser:
                waiting = self.ser.in_waiting
                if waiting:
                    data = self.ser.read(waiting)
                    self._handle_rx(data)
            elif self.bt:
                try:
                    data = self.bt.recv(1024)
                    if data:
                        self._handle_rx(data)
                except Exception:
                    pass  # 非阻塞，无数据直接忽略
            elif self.sock:
                try:
                    data = self.sock.recv(2048)
                    if data:
                        self._handle_rx(data)
                except Exception:
                    pass
        except Exception as e:
            self.info.emit(f"接收异常：{e}")
            self._log.error("RX error: %s", e)

    def _handle_rx(self, data: bytes):
        """分行解析 + 发射 device_feedback"""
        try:
            self._rx_buffer.extend(data)
            while b"\n" in self._rx_buffer:
                line, _, rest = self._rx_buffer.partition(b"\n")
                self._rx_buffer = bytearray(rest)
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    text = repr(bytes(line))
                if text:
                    # 对常见 ACK/NACK 做一下提示
                    if text.lower() in ("ok", "ack", "success"):
                        self.info.emit(f"设备应答：{text}")
                        self._log.info("Device ACK: %s", text)
                    elif text.lower() in ("err", "error", "nack", "fail"):
                        self.info.emit(f"设备应答：{text}")
                        self._log.warning("Device NACK: %s", text)
                    else:
                        self._log.info("Device>> %s", text)
                    self.device_feedback.emit(text)
        except Exception as e:
            self.info.emit(f"解析异常：{e}")
            self._log.error("Parse error: %s", e)
