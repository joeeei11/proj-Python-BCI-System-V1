# -*- coding: utf-8 -*-
# dashboard_module.py
#
# 主界面与控制模块（总控面板）
# 功能：
#   1) 主界面展示：当前用户、当前任务、实时脑电波图、阶段进度、上一次预测/外设反馈
#   2) 任务管理：开始 / 停止 训练（调用 TaskModule）
#   3) 实时脑电波：接收 EEG 数据实时绘制（支持演示模式）
#   4) 用户操作与设备控制：快速发送左/右指令、连接/断开外设（调用 ControlPanel）
#
# 和现有模块的对接方式（建议按下述 bind_* 方法在 main.py 里连接）：
#   - bind_task_module(task_page)           # 绑定 TaskModule
#   - bind_eeg_module(eeg_page)             # 绑定 EEGModule（接 trial_result 等）
#   - bind_device_control(device_page)      # 绑定 ControlPanel（接设备反馈、控制外设）
#
# 依赖：
#   pip install pyqt5 pyqtgraph   #（推荐）pyqtgraph 绘图更丝滑；若未安装会自动回退到 Matplotlib
#   pip install matplotlib numpy
#
# 注意：
#   1) 如果没有实时 EEG 源，可以打开“演示EEG”产生模拟波形（便于联调 UI）
#   2) 样式：微软雅黑 + 圆角 + 苹果配色；控件尽量居中

import numpy as np
from collections import deque
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QComboBox, QStatusBar, QCheckBox, QProgressBar
)

# 优先使用 PyQtGraph 实时绘图，若不可用则回退 matplotlib
try:
    import pyqtgraph as pg
except Exception:
    pg = None

if pg is None:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

# ---- 统一风格（与全局一致） ----
APPLE_BLUE = "#007AFF"
BORDER = "#E6E6E6"
TEXT = "#323232"
YAHEI = QFont("Microsoft YaHei", 11, QFont.Bold)

def apply_style(widget: QWidget):
    widget.setFont(YAHEI)
    widget.setStyleSheet(f"""
        QWidget {{ background:#FFFFFF; color:{TEXT}; font-family:"Microsoft YaHei","微软雅黑",Arial; font-size:14px; }}
        QGroupBox {{ border:1px solid {BORDER}; border-radius:12px; padding:12px; margin-top:8px; background:#FAFAFA; font-weight:bold; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding:0 6px; }}
        QPushButton {{ background:{APPLE_BLUE}; color:#FFF; padding:10px 16px; border-radius:10px; font-weight:bold; border:none; min-width:120px; }}
        QPushButton:hover {{ background:#1A84FF; }} QPushButton:pressed {{ background:#0062CC; }}
        QPushButton:disabled {{ background:#E0E0E0; color:#9E9E9E; }}
        QComboBox {{ border:1px solid #D0D0D0; border-radius:8px; padding:6px 10px; background:#F7F7F7; min-width:160px; }}
        QComboBox:focus {{ border:1px solid {APPLE_BLUE}; background:#FFFFFF; }}
        QLabel#big {{ font-size:18px; font-weight:bold; }}
        QLabel#chip {{ background:#F0F0F0; border:1px solid {BORDER}; border-radius:14px; padding:6px 10px; min-height:28px; }}
        QLabel#chip[active="true"] {{ background:#E8F1FF; color:{APPLE_BLUE}; border:1px solid #BFD8FF; }}
    """)

class DashboardPage(QWidget):
    """主界面与控制模块（作为一个标签页加入主窗口）"""

    # 对外信号（如果你想让 main.py 或其它模块监听）
    info = pyqtSignal(str)              # 状态提示
    request_start_trial = pyqtSignal()  # 请求 TaskModule 开始一次试次
    request_abort_trial = pyqtSignal()  # 请求 TaskModule 终止试次
    quick_send = pyqtSignal(str)        # 请求外设发送 "left"/"right"
    request_connect_device = pyqtSignal()  # 请求外设连接
    request_disconnect_device = pyqtSignal() # 请求外设断开

    def __init__(self, username: str = "未命名用户"):
        super().__init__()
        self.username = username
        apply_style(self)

        # 运行参数
        self.demo_eeg = False
        self.fs = 250          # 采样率（演示用）
        self.win_sec = 8       # 显示窗口秒数（滚动缓冲）
        self.n_channels = 6    # 演示通道数（C3,Cz,C4,CP3,CPz,CP4）
        self.buf_len = self.fs * self.win_sec
        self.buffers = [deque(maxlen=self.buf_len) for _ in range(self.n_channels)]
        for ch in self.buffers:
            ch.extend([0.0]*self.buf_len)

        self._task_module = None
        self._eeg_module = None
        self._device_page = None

        self._build_ui()
        self._build_plot()

        # 定时器：刷新图像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)  # ~25 FPS

        # 定时器：演示EEG
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self._demo_step)

    # ========= UI 组装 =========
    def _build_ui(self):
        # 顶部信息条
        header_box = QGroupBox("当前信息")
        hg = QGridLayout()
        self.lab_user = QLabel(f"用户：{self.username}"); self.lab_user.setObjectName("big")
        self.lab_task = QLabel("任务：未选择"); self.lab_task.setObjectName("big")
        self.lab_stage = QLabel("环节：—"); self.lab_stage.setObjectName("chip")
        self.lab_result = QLabel("上次预测：—"); self.lab_result.setObjectName("chip")
        self.lab_device = QLabel("设备状态：未连接"); self.lab_device.setObjectName("chip")
        r=0
        hg.addWidget(self.lab_user, r,0); hg.addWidget(self.lab_task, r,1); r+=1
        hg.addWidget(QLabel("状态与反馈"), r,0)
        st = QHBoxLayout()
        st.addWidget(self.lab_stage); st.addWidget(self.lab_result); st.addWidget(self.lab_device); st.addStretch(1)
        hg.addLayout(st, r,1); r+=1
        header_box.setLayout(hg)

        # 中部：实时脑电波图
        plot_box = QGroupBox("实时脑电波（多通道叠加）")
        pv = QVBoxLayout()
        self.plot_container = QWidget()
        pv.addWidget(self.plot_container)
        plot_box.setLayout(pv)

        # 下部：控制区
        control_box = QGroupBox("训练与设备控制")
        cg = QGridLayout()
        # 任务切换（与 TaskModule 保持一致：0=左手抓握，1=右手抓握）
        self.cmb_task = QComboBox(); self.cmb_task.addItems(["左手抓握","右手抓握"])
        self.btn_start = QPushButton("开始试次")
        self.btn_stop  = QPushButton("停止试次")
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)

        # 外设快速控制
        self.btn_conn = QPushButton("连接外设")
        self.btn_disc = QPushButton("断开外设")
        self.btn_sendL= QPushButton("发送：左手(L)")
        self.btn_sendR= QPushButton("发送：右手(R)")
        self.btn_conn.clicked.connect(lambda: self.request_connect_device.emit())
        self.btn_disc.clicked.connect(lambda: self.request_disconnect_device.emit())
        self.btn_sendL.clicked.connect(lambda: self._quick("left"))
        self.btn_sendR.clicked.connect(lambda: self._quick("right"))

        # 进度与演示
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        self.chk_demo = QCheckBox("演示EEG（无设备时可勾选）")
        self.chk_demo.toggled.connect(self._toggle_demo)

        r=0
        cg.addWidget(QLabel("当前任务"), r,0); cg.addWidget(self.cmb_task, r,1);
        cg.addWidget(self.btn_start, r,2); cg.addWidget(self.btn_stop, r,3); r+=1
        cg.addWidget(QLabel("外设控制"), r,0); cg.addWidget(self.btn_conn, r,1); cg.addWidget(self.btn_disc, r,2)
        cg.addWidget(self.btn_sendL, r,3); cg.addWidget(self.btn_sendR, r,4); r+=1
        cg.addWidget(QLabel("任务进度"), r,0); cg.addWidget(self.progress, r,1,1,4); r+=1
        cg.addWidget(self.chk_demo, r,0,1,2)
        control_box.setLayout(cg)

        root = QVBoxLayout()
        root.addWidget(header_box)
        root.addWidget(plot_box, 3)
        root.addWidget(control_box)
        self.setLayout(root)

    def _build_plot(self):
        if pg is not None:
            # 使用 PyQtGraph
            self.pg_plot = pg.PlotWidget(self.plot_container)
            lay = QVBoxLayout(self.plot_container); lay.setContentsMargins(0,0,0,0)
            lay.addWidget(self.pg_plot)
            self.pg_plot.setLabel('bottom','时间 (s)')
            self.pg_plot.setLabel('left','电位 (μV)  [已垂直偏移叠加]')
            self.pg_curves = []
            for i in range(self.n_channels):
                curve = self.pg_plot.plot(pen=pg.mkPen(width=1))
                self.pg_curves.append(curve)
        else:
            # 回退 matplotlib
            self.pg_plot = None
            self.fig = Figure(figsize=(8,3), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            lay = QVBoxLayout(self.plot_container); lay.setContentsMargins(0,0,0,0)
            lay.addWidget(self.canvas)
            self.ax = self.fig.gca()
            self.lines = []
            t = np.linspace(-self.win_sec, 0, self.buf_len)
            for i in range(self.n_channels):
                line, = self.ax.plot(t, np.zeros_like(t), linewidth=0.8)
                self.lines.append(line)
            self.ax.set_xlabel("时间 (s)"); self.ax.set_ylabel("电位 (μV)")
            self.ax.set_title("EEG 实时波形")

    # ========= 外部绑定 =========
    def bind_task_module(self, task_page):
        """与 TaskModule 对接：读取当前任务、调用开始/停止、接收阶段变化"""
        self._task_module = task_page
        # 同步任务选择
        self.cmb_task.currentIndexChanged.connect(lambda idx: self._sync_task_selection(idx))
        # 从 TaskModule 收阶段变化（可直接提升到顶部状态条）
        task_page.stage.connect(self.on_stage_changed)
        task_page.info.connect(lambda s: self.info.emit(s))

    def bind_eeg_module(self, eeg_page):
        """与 EEGModule 对接：试次结果、状态提示；也可从这里订阅实时片段"""
        self._eeg_module = eeg_page
        eeg_page.trial_result.connect(self.on_trial_result)
        eeg_page.info.connect(lambda s: self.info.emit(s))

    def bind_device_control(self, device_page):
        """与 ControlPanel 对接：连接/断开、快速发送；接收设备反馈与发送结果"""
        self._device_page = device_page
        # 连接按钮动作
        self.request_connect_device.connect(lambda: self._safe_call(device_page, "连接", getattr(device_page, "btn_conn", None)))
        self.request_disconnect_device.connect(lambda: self._safe_call(device_page, "断开", getattr(device_page, "btn_disc", None)))
        self.quick_send.connect(lambda lab: self._quick_to_device(device_page, lab))
        # 反馈联动
        device_page.device_feedback.connect(self.on_device_feedback)
        device_page.send_result.connect(self.on_device_send_result)
        device_page.info.connect(lambda s: self.info.emit(s))

    # ========= 事件处理 =========
    def _sync_task_selection(self, idx: int):
        """用户在总控面板切换任务时，同步 TaskModule 的任务下拉框"""
        if self._task_module is None: return
        try:
            if getattr(self._task_module, "task", None):
                self._task_module.task.setCurrentIndex(idx)
            self.lab_task.setText(f"任务：{'左手抓握' if idx==0 else '右手抓握'}")
        except Exception:
            pass

    def _start_clicked(self):
        # 主面板 → 请求 TaskModule 开始一次试次
        if self._task_module and hasattr(self._task_module, "start_trial"):
            self._task_module.start_trial()
            self.info.emit("开始试次")
        else:
            self.info.emit("未绑定任务模块，无法开始试次")

    def _stop_clicked(self):
        # 主面板 → 请求 TaskModule 停止试次
        if self._task_module and hasattr(self._task_module, "abort_trial"):
            self._task_module.abort_trial()
            self.info.emit("已停止试次")
        else:
            self.info.emit("未绑定任务模块，无法停止试次")

    def _quick(self, label: str):
        """快速给外设下发 L/R 指令"""
        if label not in ("left","right"): return
        self.quick_send.emit(label)

    def _quick_to_device(self, device_page, label: str):
        try:
            if hasattr(device_page, "send_by_label"):
                device_page.send_by_label(label)
                self.info.emit(f"外设：已发送 {label}")
        except Exception as e:
            self.info.emit(f"外设发送异常：{e}")

    def _safe_call(self, device_page, name, btn):
        """点击 device_page 的连接/断开按钮（保证不崩）"""
        try:
            if btn is not None: btn.click()
        except Exception as e:
            self.info.emit(f"{name}失败：{e}")

    # ========= 与其它模块联动的槽 =========
    def on_stage_changed(self, stage_name: str, idx: int):
        """来自 TaskModule 的阶段变化：更新芯片标签 + 进度条"""
        self.lab_stage.setText(f"环节：{stage_name}")
        self.lab_stage.setProperty("active", True)
        self.lab_stage.style().unpolish(self.lab_stage); self.lab_stage.style().polish(self.lab_stage)
        # 简单映射进度（0..3）→ 进度条
        self.progress.setValue(int((idx+1)/4*100))

        # 注视点阶段：清空上次预测显示
        if stage_name == "注视点":
            self.lab_result.setText("上次预测：—")

    def on_trial_result(self, pred: str, success: bool):
        """来自 EEGModule 的预测结果"""
        label = "左" if pred=="left" else "右"
        tip = "成功✅" if success else "失败❌"
        self.lab_result.setText(f"上次预测：{label}（{tip}）")

    def on_device_send_result(self, ok: bool, message: str):
        self.lab_device.setText(f"设备状态：{'发送成功' if ok else '发送失败'} | {message}")

    def on_device_feedback(self, feedback: str):
        self.lab_device.setText(f"设备状态：反馈 {feedback}")

    # ========= EEG 数据通路 =========
    def feed_eeg_samples(self, values):
        """
        外部推入一帧 EEG 样本（多通道），如 values = [C3,Cz,C4,CP3,CPz,CP4]
        - 若长度与通道数不一致，将自动裁剪/填0
        - 你可以从串口/蓝牙的接收线程解析后调用此函数
        """
        if not isinstance(values, (list, tuple, np.ndarray)): return
        vals = list(values)[:self.n_channels]
        if len(vals) < self.n_channels:
            vals += [0.0]*(self.n_channels - len(vals))
        for i,v in enumerate(vals):
            self.buffers[i].append(float(v))

    def _toggle_demo(self, on: bool):
        self.demo_eeg = on
        if on:
            self.demo_phase = 0.0
            self.demo_timer.start(20)
            self.info.emit("演示EEG：已开启")
        else:
            self.demo_timer.stop()
            self.info.emit("演示EEG：已关闭")

    def _demo_step(self):
        """简单的多通道合成波（方便无设备时联调UI）"""
        t = self.demo_phase
        # 左/右手“节律”差异（仅用于演示）
        base = np.sin(2*np.pi*10*(t)) * 10.0          # ~10Hz μ节律
        drift= np.sin(2*np.pi*0.3*(t)) * 3.0          # 低频漂移
        noise= np.random.randn(self.n_channels) * 0.8 # 噪声
        sample = base + drift + noise
        # 构造六通道：稍微变化相位
        vals = [sample + 2.0*np.sin(2*np.pi*0.7*t + i) for i in range(self.n_channels)]
        # 推入缓冲
        for i in range(self.n_channels):
            self.buffers[i].append(float(vals[i]))
        self.demo_phase += 1.0/self.fs

    # ========= 绘图刷新 =========
    def _tick(self):
        if pg is not None:
            # 使用 PyQtGraph：高性能滚动刷新
            t = np.linspace(-self.win_sec, 0, self.buf_len)
            shift = 0.0
            for i,curve in enumerate(self.pg_curves):
                y = np.array(self.buffers[i])
                if y.size < self.buf_len:
                    y = np.pad(y, (self.buf_len - y.size, 0), mode='constant')
                curve.setData(t, y + shift)
                shift += max(1.0, np.nanmax(np.abs(y))*1.2 + 10.0)
        else:
            # 回退 matplotlib
            t = np.linspace(-self.win_sec, 0, self.buf_len)
            shift = 0.0
            for i,line in enumerate(self.lines):
                y = np.array(self.buffers[i])
                if y.size < self.buf_len:
                    y = np.pad(y, (self.buf_len - y.size, 0), mode='constant')
                line.set_data(t, y + shift)
                shift += max(1.0, np.nanmax(np.abs(y))*1.2 + 10.0)
            self.ax.relim(); self.ax.autoscale_view()
            self.canvas.draw()
