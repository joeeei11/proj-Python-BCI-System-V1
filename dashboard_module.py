# -*- coding: utf-8 -*-
# dashboard_module.py
# 仪表盘模块 (Phase 20 Step 2: Visual Config & Channel Filtering)
# 职责：实时波形显示、系统状态概览、快捷控制、通道可视化配置
# 优化：支持通道重命名、动态筛选、自动堆叠显示

import numpy as np
from collections import deque

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QGridLayout, QCheckBox
)

# --- Fluent Widgets ---
from qfluentwidgets import (
    CardWidget, SimpleCardWidget, ElevatedCardWidget,
    PrimaryPushButton, PushButton, ComboBox, ProgressBar, ToggleButton, LineEdit,
    TitleLabel, SubtitleLabel, CaptionLabel, StrongBodyLabel, BodyLabel,
    InfoBadge, InfoLevel, IconWidget, FluentIcon as FIF,
    setTheme, Theme
)

# --- Plotting Libs ---
try:
    import pyqtgraph as pg

    # 适配 Light 主题配置
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOption('antialias', True)  # 开启抗锯齿
except ImportError:
    pg = None
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure


class DashboardPage(QWidget):
    """主界面 (Dashboard) - 实时绘图优化版"""

    # --- 信号定义 ---
    info = pyqtSignal(str)
    request_start_trial = pyqtSignal()
    request_abort_trial = pyqtSignal()
    quick_send = pyqtSignal(str)
    request_connect_device = pyqtSignal()
    request_disconnect_device = pyqtSignal()

    def __init__(self, username: str = "用户"):
        super().__init__()
        self.username = username

        # --- 基础配置 ---
        self.fs = 250  # 默认采样率 (会被 Worker 覆盖)
        self.win_sec = 5  # 显示窗口长度 (秒)
        self.n_channels = 8  # 默认支持的最大通道数
        self.buf_len = self.fs * self.win_sec
        self.scale_factor = 50.0  # 默认通道间距 uV

        # --- 通道可视化配置 (Phase 20) ---
        # 默认名称映射 (Standard MI Montage)
        self.default_map = "Fz, C3, Cz, C4, Pz, O1, O2, P3"
        self.ch_names = [x.strip() for x in self.default_map.split(',')]
        # 补齐名称
        while len(self.ch_names) < self.n_channels:
            self.ch_names.append(f"Ch{len(self.ch_names) + 1}")

        # 可见性状态
        self.ch_vis = [True] * self.n_channels

        # 初始化缓冲区 (预填充 0)
        self.buffers = [deque([0.0] * self.buf_len, maxlen=self.buf_len) for _ in range(self.n_channels)]

        # 内部引用
        self._task_module = None
        self._eeg_module = None
        self._device_page = None
        self.demo_eeg = False

        self._init_ui()
        self._init_chart()

        # 绘图定时器 (30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

        # 演示数据定时器
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self._demo_step)

    def _init_ui(self):
        self.v_layout = QVBoxLayout(self)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        self.v_layout.setSpacing(16)

        # =================================================
        # 1. Header Card
        # =================================================
        self.header_card = SimpleCardWidget(self)
        self.header_card.setFixedHeight(88)
        h_layout = QHBoxLayout(self.header_card)
        h_layout.setContentsMargins(24, 0, 24, 0)
        h_layout.setSpacing(16)

        icon = IconWidget(FIF.PEOPLE)
        icon.setFixedSize(40, 40)

        text_l = QVBoxLayout()
        text_l.setAlignment(Qt.AlignVCenter)
        title = TitleLabel(f"欢迎回来，{self.username}", self)
        subtitle = CaptionLabel("NeuroPilot 脑机接口康复系统 - 就绪", self)
        subtitle.setTextColor(QColor(96, 96, 96), QColor(200, 200, 200))
        text_l.addWidget(title)
        text_l.addWidget(subtitle)

        # 状态徽章
        self.badge_task = InfoBadge.info("任务: 未选择")
        self.badge_stage = InfoBadge.attension("环节: 待机")
        self.badge_device = InfoBadge.error("设备: 未连接")

        h_layout.addWidget(icon)
        h_layout.addLayout(text_l)
        h_layout.addStretch(1)
        h_layout.addWidget(self.badge_task)
        h_layout.addWidget(self.badge_stage)
        h_layout.addWidget(self.badge_device)

        # =================================================
        # 2. Chart Card (Elevated)
        # =================================================
        self.chart_card = ElevatedCardWidget(self)
        self.chart_card.setBorderRadius(10)
        chart_l = QVBoxLayout(self.chart_card)
        chart_l.setContentsMargins(16, 12, 16, 16)

        # Header
        chart_header = QHBoxLayout()
        chart_title = SubtitleLabel("实时脑电监测 (Real-time EEG)", self)

        self.btn_demo = ToggleButton(self)
        self.btn_demo.setText("演示模式")
        self.btn_demo.toggled.connect(self._toggle_demo)

        chart_header.addWidget(chart_title)
        chart_header.addStretch(1)
        chart_header.addWidget(self.btn_demo)

        # Container
        self.plot_container = QWidget()

        chart_l.addLayout(chart_header)
        chart_l.addWidget(self.plot_container, 1)

        # =================================================
        # 3. Visual Config Card (New in Phase 20)
        # =================================================
        self.vis_card = SimpleCardWidget(self)
        vis_l = QVBoxLayout(self.vis_card)
        vis_l.setContentsMargins(24, 16, 24, 16)
        vis_l.setSpacing(12)

        vis_l.addWidget(StrongBodyLabel("通道可视化配置", self))

        # Row 1: Name Mapping
        row_map = QHBoxLayout()
        row_map.addWidget(CaptionLabel("通道名称映射 (逗号分隔):", self))
        self.ed_ch_map = LineEdit(self)
        self.ed_ch_map.setText(self.default_map)
        self.ed_ch_map.setPlaceholderText("例如: Fp1, Fp2, C3, C4...")
        self.ed_ch_map.textChanged.connect(self._update_ch_config)
        row_map.addWidget(self.ed_ch_map, 1)
        vis_l.addLayout(row_map)

        # Row 2: Visibility Toggles
        row_chk = QHBoxLayout()
        row_chk.addWidget(CaptionLabel("显示通道:", self))

        self.chk_list = []
        chk_grid = QGridLayout()
        chk_grid.setContentsMargins(0, 0, 0, 0)
        chk_grid.setHorizontalSpacing(16)

        for i in range(self.n_channels):
            # 使用原生 QCheckBox 因为更紧凑，或者 Fluent CheckBox
            chk = QCheckBox(f"Ch{i + 1}")
            chk.setChecked(True)
            chk.stateChanged.connect(self._update_ch_config)
            # 简单的 2行布局
            chk_grid.addWidget(chk, 0 if i < 4 else 1, i % 4)
            self.chk_list.append(chk)

        row_chk.addLayout(chk_grid)
        row_chk.addStretch(1)
        vis_l.addLayout(row_chk)

        # =================================================
        # 4. Control Card
        # =================================================
        self.control_card = CardWidget(self)
        self.control_card.setFixedHeight(120)
        ctrl_l = QHBoxLayout(self.control_card)
        ctrl_l.setContentsMargins(24, 16, 24, 16)
        ctrl_l.setSpacing(24)

        # A. 任务
        task_l = QVBoxLayout()
        task_l.setSpacing(8)
        task_title = StrongBodyLabel("任务控制", self)
        self.cmb_task = ComboBox(self)
        self.cmb_task.addItems(["左手抓握", "右手抓握"])
        self.cmb_task.setFixedWidth(160)

        btn_box = QHBoxLayout()
        self.btn_start = PrimaryPushButton(FIF.CARE_RIGHT_SOLID, "开始", self)
        self.btn_stop = PushButton(FIF.PAUSE, "中止", self)
        self.btn_start.setFixedWidth(75)
        self.btn_stop.setFixedWidth(75)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_stop)

        task_l.addWidget(task_title)
        task_l.addWidget(self.cmb_task)
        task_l.addLayout(btn_box)

        # B. 进度
        prog_l = QVBoxLayout()
        prog_l.setSpacing(8)
        prog_title = StrongBodyLabel("当前环节进度", self)
        self.progress = ProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(280)
        self.lab_result = CaptionLabel("上次预测: —", self)

        prog_l.addWidget(prog_title)
        prog_l.addWidget(self.progress)
        prog_l.addWidget(self.lab_result)

        # C. 外设
        dev_l = QVBoxLayout()
        dev_l.setSpacing(8)
        dev_title = StrongBodyLabel("外设操作", self)

        dev_r1 = QHBoxLayout()
        self.btn_conn = PushButton(FIF.IOT, "连接", self)
        self.btn_disc = PushButton(FIF.CANCEL, "断开", self)
        dev_r1.addWidget(self.btn_conn)
        dev_r1.addWidget(self.btn_disc)

        dev_r2 = QHBoxLayout()
        self.btn_L = PushButton("发左(L)", self)
        self.btn_R = PushButton("发右(R)", self)
        dev_r2.addWidget(self.btn_L)
        dev_r2.addWidget(self.btn_R)

        self.btn_conn.clicked.connect(lambda: self.request_connect_device.emit())
        self.btn_disc.clicked.connect(lambda: self.request_disconnect_device.emit())
        self.btn_L.clicked.connect(lambda: self._quick("left"))
        self.btn_R.clicked.connect(lambda: self._quick("right"))

        dev_l.addWidget(dev_title)
        dev_l.addLayout(dev_r1)
        dev_l.addLayout(dev_r2)

        # Layout Assembly
        ctrl_l.addLayout(task_l)
        line1 = QFrame()
        line1.setFrameShape(QFrame.VLine)
        line1.setStyleSheet("color: #E5E5E5;")
        ctrl_l.addWidget(line1)

        ctrl_l.addLayout(prog_l)
        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setStyleSheet("color: #E5E5E5;")
        ctrl_l.addWidget(line2)

        ctrl_l.addLayout(dev_l)
        ctrl_l.addStretch(1)

        # Final Vertical Stack
        self.v_layout.addWidget(self.header_card)
        self.v_layout.addWidget(self.chart_card, 1)
        self.v_layout.addWidget(self.vis_card)  # Inserted
        self.v_layout.addWidget(self.control_card)

        # Trigger initial update
        self._update_ch_config()

    def _init_chart(self):
        layout = QVBoxLayout(self.plot_container)
        layout.setContentsMargins(0, 0, 0, 0)

        if pg:
            self.pg_plot = pg.PlotWidget()
            # 视觉微调：纯白背景，灰色网格
            self.pg_plot.setBackground('#FFFFFF')
            self.pg_plot.showGrid(x=True, y=True, alpha=0.15)
            self.pg_plot.getViewBox().setBorder(None)

            # 交互增强：启用鼠标缩放和平移
            self.pg_plot.setMouseEnabled(x=True, y=True)
            self.pg_plot.enableAutoRange()

            self.pg_plot.setLabel('bottom', 'Time (s)', color='#666666')
            # Y Label 留空，通过 Axis Ticks 显示通道名

            layout.addWidget(self.pg_plot)

            self.pg_curves = []
            # 现代配色
            colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#607D8B', '#E91E63']
            for i in range(self.n_channels):
                # 稍微加粗一点
                pen = pg.mkPen(color=colors[i % len(colors)], width=1.2)
                curve = self.pg_plot.plot(pen=pen)
                self.pg_curves.append(curve)
        else:
            # Matplotlib Fallback
            self.pg_plot = None
            self.fig = Figure(figsize=(8, 3), tight_layout=True)
            self.fig.patch.set_facecolor('#FFFFFF')
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas)
            self.ax = self.fig.gca()
            self.lines = []
            t = np.linspace(-self.win_sec, 0, self.buf_len)
            for i in range(self.n_channels):
                line, = self.ax.plot(t, np.zeros_like(t), linewidth=1)
                self.lines.append(line)
            self.ax.set_xlabel("Time (s)")
            self.ax.grid(True, alpha=0.2)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)

    # ======================================================
    # Logic & Signals
    # ======================================================

    def _update_ch_config(self):
        """解析用户配置的通道名称与可见性"""
        # 1. 解析名称
        text = self.ed_ch_map.text()
        names = [x.strip() for x in text.split(',') if x.strip()]
        # 补齐
        while len(names) < self.n_channels:
            names.append(f"Ch{len(names) + 1}")
        self.ch_names = names[:self.n_channels]

        # 2. 更新 Checkbox 文本与状态
        for i, chk in enumerate(self.chk_list):
            if i < len(self.ch_names):
                chk.setText(self.ch_names[i])
                self.ch_vis[i] = chk.isChecked()
            else:
                self.ch_vis[i] = False

    def bind_task_module(self, task_page):
        self._task_module = task_page
        self.cmb_task.currentIndexChanged.connect(lambda idx: self._sync_task(idx))
        task_page.stage.connect(self.on_stage_changed)
        task_page.info.connect(lambda s: self.info.emit(s))

    def bind_eeg_module(self, eeg_page):
        self._eeg_module = eeg_page
        eeg_page.raw_data_ready.connect(self.feed_eeg_samples)
        eeg_page.trial_result.connect(self.on_trial_result)
        eeg_page.info.connect(lambda s: self.info.emit(s))

    def bind_device_control(self, device_page):
        self._device_page = device_page
        self.request_connect_device.connect(lambda: self._safe_click(getattr(device_page, "btn_connect", None)))
        self.request_disconnect_device.connect(lambda: self._safe_click(getattr(device_page, "btn_disconnect", None)))
        self.quick_send.connect(lambda lab: self._quick_dev(device_page, lab))
        device_page.device_feedback.connect(self.on_device_feedback)
        device_page.send_result.connect(self.on_device_send_result)
        device_page.info.connect(lambda s: self.info.emit(s))

    def _sync_task(self, idx):
        if self._task_module and hasattr(self._task_module, "task"):
            self._task_module.task.setCurrentIndex(idx)
        name = "左手" if idx == 0 else "右手"
        self.badge_task.setText(f"任务: {name}")

    def _start_clicked(self):
        if self._task_module and hasattr(self._task_module, "start_trial"):
            self._task_module.start_trial()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

    def _stop_clicked(self):
        if self._task_module and hasattr(self._task_module, "abort_trial"):
            self._task_module.abort_trial()

    def _quick(self, label):
        self.quick_send.emit(label)

    def _quick_dev(self, page, label):
        if hasattr(page, "_send_cmd"):
            page._send_cmd(label)

    def _safe_click(self, btn):
        if btn: btn.click()

    # --- 状态更新 ---

    def on_stage_changed(self, stage: str, idx: int):
        self.badge_stage.setText(f"环节: {stage}")
        if stage == "运动想象":
            try:
                self.badge_stage.setLevel(InfoLevel.WARNING)
            except:
                pass
        elif stage == "休息":
            try:
                self.badge_stage.setLevel(InfoLevel.SUCCESS)
            except:
                pass

        progress = int((idx + 1) / 4.0 * 100)
        self.progress.setValue(progress)

        if stage in ["休息结束", "已中止"]:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def on_trial_result(self, pred, success):
        icon = "✅" if success else "❌"
        label = "左手" if pred == "left" else ("右手" if pred == "right" else "未知")
        self.lab_result.setText(f"上次预测: {label} {icon}")

    def on_device_send_result(self, ok, msg):
        self.badge_device.setText("发送成功" if ok else "发送失败")
        try:
            self.badge_device.setLevel(InfoLevel.SUCCESS if ok else InfoLevel.ERROR)
        except:
            pass

    def on_device_feedback(self, msg):
        self.badge_device.setText(f"反馈: {msg}")

    # ======================================================
    # Data Ingestion & Plotting (关键优化)
    # ======================================================

    def feed_eeg_samples(self, values):
        """
        接收 EEG 数据块。
        Args:
            values: np.ndarray shape (n_samples, n_channels)
        """
        if self.demo_eeg:
            self.btn_demo.setChecked(False)  # 触发 _toggle_demo(False)

        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.ndim == 1:
            values = values.reshape(1, -1)

        n_samples, n_ch_in = values.shape

        limit = min(self.n_channels, n_ch_in)
        for i in range(limit):
            self.buffers[i].extend(values[:, i])

    def _tick(self):
        """定时刷新绘图 (支持通道动态筛选与堆叠)"""
        if not self.isVisible(): return

        # 1. 获取当前需要显示的通道索引
        visible_indices = [i for i, vis in enumerate(self.ch_vis) if vis]

        # 2. 绘图
        if pg:
            t = np.linspace(-self.win_sec, 0, self.buf_len)
            ticks = []  # 用于 Y 轴标签 [(value, label), ...]

            # 遍历所有曲线，分别处理可见与不可见
            for i, curve in enumerate(self.pg_curves):
                if i in visible_indices:
                    # 计算堆叠偏移量：根据它在 visible_indices 中的顺序 (stack_idx)
                    stack_idx = visible_indices.index(i)
                    offset = stack_idx * self.scale_factor

                    data = np.array(self.buffers[i])
                    # 去直流
                    if len(data) > 0:
                        data = data - np.mean(data)
                    # 补齐
                    if len(data) < self.buf_len:
                        pad = np.zeros(self.buf_len - len(data))
                        data = np.concatenate([pad, data])

                    # 绘制
                    curve.setData(t, data + offset)
                    curve.setVisible(True)

                    # 记录 Y 轴标签位置
                    ticks.append((offset, self.ch_names[i]))
                else:
                    # 隐藏不可见通道
                    curve.setData([], [])
                    curve.setVisible(False)

            # 更新 Y 轴标签 (只显示可见通道的名称)
            # setTicks 需要传入 list of list: [ [(val, text), ...], [] ]
            self.pg_plot.getPlotItem().getAxis('left').setTicks([ticks])

        else:
            # Matplotlib Fallback
            t = np.linspace(-self.win_sec, 0, self.buf_len)
            yticks = []
            yticklabels = []

            for i, line in enumerate(self.lines):
                if i in visible_indices:
                    stack_idx = visible_indices.index(i)
                    offset = stack_idx * self.scale_factor

                    data = np.array(self.buffers[i])
                    if len(data) > 0: data = data - np.mean(data)
                    if len(data) < self.buf_len:
                        data = np.pad(data, (self.buf_len - len(data), 0), mode='edge')

                    line.set_data(t, data + offset)
                    line.set_visible(True)

                    yticks.append(offset)
                    yticklabels.append(self.ch_names[i])
                else:
                    line.set_visible(False)

            # 更新坐标轴
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels(yticklabels)

            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()

    def _toggle_demo(self, on: bool):
        self.demo_eeg = on
        if on:
            self.demo_phase = 0.0
            self.demo_timer.start(20)
            self.btn_demo.setText("停止演示")
        else:
            self.demo_timer.stop()
            self.btn_demo.setText("演示模式")

    def _demo_step(self):
        t = self.demo_phase
        vals = []
        for i in range(self.n_channels):
            base = np.sin(2 * np.pi * 10 * t) * 10.0
            noise = np.random.randn() * 2.0
            vals.append(base + noise + i * 10)

        for i in range(self.n_channels):
            self.buffers[i].extend([vals[i]])

        self.demo_phase += 1.0 / self.fs