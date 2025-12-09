# -*- coding: utf-8 -*-
# debug_module.py
# 全能调试模块 (Fluent Design Phase 14)
# 职责：实时监视硬件流量、调试指令发送、协议分析

import binascii
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
)

# --- Fluent Widgets ---
from qfluentwidgets import (
    SmoothScrollArea, CardWidget, SimpleCardWidget,
    PlainTextEdit, LineEdit, PrimaryPushButton, PushButton,
    SwitchButton, CheckBox, ToolButton,
    TitleLabel, SubtitleLabel, BodyLabel, CaptionLabel, StrongBodyLabel,
    FluentIcon as FIF, IconWidget, InfoBar
)


class DebugPanel(QWidget):
    """
    硬件调试控制台
    """
    # 请求向外设发送数据 (bytes)
    request_send_device = pyqtSignal(bytes)
    info = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DebugPanel")

        self._paused = False
        self._init_ui()

    def _init_ui(self):
        # 1. 根布局：SmoothScrollArea
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = SmoothScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")

        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)

        # 2. 内容布局
        self.v_layout = QVBoxLayout(self.content_widget)
        self.v_layout.setContentsMargins(24, 24, 24, 24)
        self.v_layout.setSpacing(20)

        # ==========================================
        # Header
        # ==========================================
        self.header = SimpleCardWidget()
        h_header = QHBoxLayout(self.header)
        h_header.setContentsMargins(24, 16, 24, 16)

        icon = IconWidget(FIF.DEVELOPER_TOOLS)
        icon.setFixedSize(40, 40)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(TitleLabel("系统调试控制台", self))
        title_box.addWidget(CaptionLabel("实时监视 Device/EEG 数据流，发送底层指令", self))

        # 全局控制
        self.btn_pause = PushButton(FIF.PAUSE, "暂停滚动", self)
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._toggle_pause)

        self.btn_clear = PushButton(FIF.DELETE, "清空日志", self)
        self.btn_clear.clicked.connect(self._clear_logs)

        h_header.addWidget(icon)
        h_header.addLayout(title_box)
        h_header.addStretch(1)
        h_header.addWidget(self.btn_pause)
        h_header.addWidget(self.btn_clear)

        self.v_layout.addWidget(self.header)

        # ==========================================
        # Log Area (Split View)
        # ==========================================
        log_layout = QHBoxLayout()
        log_layout.setSpacing(16)

        # --- Left: Device Logs ---
        self.dev_card = CardWidget()
        l_dev = QVBoxLayout(self.dev_card)
        l_dev.setContentsMargins(16, 12, 16, 16)

        dev_header = QHBoxLayout()
        dev_header.addWidget(StrongBodyLabel("🎮 外设通讯 (Device)", self))
        dev_header.addStretch(1)
        self.chk_dev_hex = CheckBox("HEX 显示")
        dev_header.addWidget(self.chk_dev_hex)

        self.txt_dev = PlainTextEdit()
        self.txt_dev.setReadOnly(True)
        self.txt_dev.setFont(QFont("Consolas", 9))
        self.txt_dev.setPlaceholderText("等待外设数据...")

        l_dev.addLayout(dev_header)
        l_dev.addWidget(self.txt_dev)
        log_layout.addWidget(self.dev_card, 1)

        # --- Right: EEG Logs ---
        self.eeg_card = CardWidget()
        l_eeg = QVBoxLayout(self.eeg_card)
        l_eeg.setContentsMargins(16, 12, 16, 16)

        eeg_header = QHBoxLayout()
        eeg_header.addWidget(StrongBodyLabel("🧠 脑机数据流 (EEG)", self))
        eeg_header.addStretch(1)

        self.txt_eeg = PlainTextEdit()
        self.txt_eeg.setReadOnly(True)
        self.txt_eeg.setFont(QFont("Consolas", 9))
        self.txt_eeg.setPlaceholderText("等待 EEG 采集流...")

        l_eeg.addLayout(eeg_header)
        l_eeg.addWidget(self.txt_eeg)
        log_layout.addWidget(self.eeg_card, 1)

        self.v_layout.addLayout(log_layout, 1)  # 拉伸权重 1

        # ==========================================
        # Control Area (Device Send)
        # ==========================================
        self.ctrl_card = SimpleCardWidget()
        l_ctrl = QHBoxLayout(self.ctrl_card)
        l_ctrl.setContentsMargins(24, 16, 24, 16)
        l_ctrl.setSpacing(16)

        l_ctrl.addWidget(IconWidget(FIF.COMMAND_PROMPT))
        l_ctrl.addWidget(StrongBodyLabel("指令发送", self))

        self.ed_cmd = LineEdit()
        self.ed_cmd.setPlaceholderText("输入指令 (例如: L 或 4C 0A)")
        self.ed_cmd.returnPressed.connect(self._on_send)

        self.sw_hex_send = SwitchButton("HEX 模式")
        self.sw_hex_send.setOnText("HEX")
        self.sw_hex_send.setOffText("ASCII")

        self.btn_send = PrimaryPushButton(FIF.SEND, "发送", self)
        self.btn_send.clicked.connect(self._on_send)

        l_ctrl.addWidget(self.ed_cmd, 1)
        l_ctrl.addWidget(self.sw_hex_send)
        l_ctrl.addWidget(self.btn_send)

        self.v_layout.addWidget(self.ctrl_card)

    # ==========================================
    # Logic Slots
    # ==========================================

    @pyqtSlot(str, object)
    def append_device_log(self, direction, data):
        """
        Device Log Slot
        direction: "TX" / "RX" / "INFO"
        data: bytes or str
        """
        if self._paused: return

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 格式化内容
        content = ""
        if isinstance(data, bytes):
            if self.chk_dev_hex.isChecked():
                content = data.hex(' ').upper()
            else:
                try:
                    content = data.decode('utf-8', errors='replace').strip()
                except:
                    content = str(data)
        else:
            content = str(data)

        # 颜色标记 (HTML)
        color = "#333333"  # Default black
        if direction == "TX":
            color = "#009FAA"  # Teal
        elif direction == "RX":
            color = "#E91E63"  # Pink
        elif direction == "INFO":
            color = "#FF9800"  # Orange

        log_line = f'<span style="color:#999;">[{ts}]</span> <b style="color:{color}">{direction}:</b> {content}'
        self.txt_dev.appendHtml(log_line)

    @pyqtSlot(str, object)
    def append_eeg_log(self, direction, data):
        """
        EEG Log Slot
        """
        if self._paused: return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        content = str(data)
        color = "#333333"
        if direction == "INFO":
            color = "#2196F3"  # Blue
        elif direction == "RX":
            color = "#4CAF50"  # Green

        log_line = f'<span style="color:#999;">[{ts}]</span> <b style="color:{color}">{direction}:</b> {content}'
        self.txt_eeg.appendHtml(log_line)

    def _on_send(self):
        text = self.ed_cmd.text().strip()
        if not text: return

        payload = b''
        try:
            if self.sw_hex_send.isChecked():
                # Remove spaces and parse hex
                clean_hex = text.replace(" ", "")
                payload = binascii.unhexlify(clean_hex)
            else:
                # ASCII mode
                # Auto append newline if missing? let's keep it raw for debug
                payload = text.encode('utf-8')

                # Support C-style escapes
                if "\\n" in text: payload = payload.decode('string_escape').encode(
                    'utf-8') if bytes is str else text.encode('utf-8').replace(b'\\n', b'\n').replace(b'\\r', b'\r')

            self.request_send_device.emit(payload)
            self.ed_cmd.clear()

        except Exception as e:
            InfoBar.error("发送错误", str(e), parent=self)

    def _toggle_pause(self, checked):
        self._paused = checked
        self.btn_pause.setText("继续滚动" if checked else "暂停滚动")

    def _clear_logs(self):
        self.txt_dev.clear()
        self.txt_eeg.clear()