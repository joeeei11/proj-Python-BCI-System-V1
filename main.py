# -*- coding: utf-8 -*-
# main.py
#
# 标签页：
#   1) （可选）主界面 Dashboard（若 dashboard_module 存在）
#   2) 运动范式（task_module.TaskModule）
#   3) 脑电分类（eeg_module.EEGModule）
#   4) 外设控制（device_control.ControlPanel）
#   5) 模型训练与优化（ml_module.MLTrainerPanel）← 新增
#   6) （可选）系统日志（若 log_module.LogPanel 存在）

import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QStatusBar, QDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# —— 必选模块（你项目中已有）——
from login_dialog import LoginDialog
from task_module import TaskModule
from eeg_module import EEGModule
from device_control import ControlPanel

# —— 新增：训练与优化面板（请确保有 ml_module.py）——
try:
    from ml_module import MLTrainerPanel
except Exception:
    MLTrainerPanel = None

# —— 可选：主界面与日志页（存在就加载，不存在不影响）——
try:
    from dashboard_module import DashboardPage
except Exception:
    DashboardPage = None

try:
    from log_module import LogPanel
except Exception:
    LogPanel = None


class MainWindow(QMainWindow):
    def __init__(self, username: str):
        super().__init__()
        self.setWindowTitle("NeuroPilot 运动想象上肢康复系统")
        self.resize(1200, 800)
        self.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))

        # —— logger：供系统日志页接入（若存在）——
        self._log = logging.getLogger("NeuroPilot")
        if not self._log.handlers:
            self._log.setLevel(logging.INFO)
            sh = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
            sh.setFormatter(fmt)
            self._log.addHandler(sh)

        # —— 顶部用户信息 ——
        self.header = QLabel(f"当前用户：{username}")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("font-size:16px; padding:8px;")

        # —— 标签容器 ——
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabBar::tab { padding: 10px 18px; font-weight: bold; }
            QTabBar::tab:selected { color: #007AFF; }
        """)

        # —— 页面实例化（保持原结构命名）——
        self.task_page = TaskModule()      # 运动范式
        self.eeg_page = EEGModule()        # 脑电分类
        self.device_page = ControlPanel()  # 外设控制

        self.dashboard_page = DashboardPage(username) if DashboardPage else None
        self.log_page = LogPanel() if LogPanel else None
        self.ml_page = MLTrainerPanel() if MLTrainerPanel else None  # 新增页

        # —— 信号联动（保持原有 + 新增行）——
        self.task_page.info.connect(self.on_info)
        self.task_page.stage.connect(self.on_stage)
        self.task_page.trial_progress.connect(self.on_progress)

        self.eeg_page.info.connect(self.on_info)
        self.eeg_page.trial_result.connect(self.on_trial_result)

        self.device_page.info.connect(self.on_info)
        # 设备即时反馈（如 ACK/ERR）
        if hasattr(self.device_page, "device_feedback"):
            self.device_page.device_feedback.connect(self.on_device_feedback)
        # 发送结果（成功/失败）回灌
        if hasattr(self.device_page, "send_result"):
            self.device_page.send_result.connect(self.on_device_send_result)

        # 可选页面信号绑定
        if self.dashboard_page and hasattr(self.dashboard_page, "info"):
            self.dashboard_page.info.connect(self.on_info)
            # 如你的 DashboardPage 支持与其他模块绑定，保持原有绑定方式：
            if hasattr(self.dashboard_page, "bind_task_module"):
                self.dashboard_page.bind_task_module(self.task_page)
            if hasattr(self.dashboard_page, "bind_eeg_module"):
                self.dashboard_page.bind_eeg_module(self.eeg_page)
            if hasattr(self.dashboard_page, "bind_device_control"):
                self.dashboard_page.bind_device_control(self.device_page)

        if self.ml_page and hasattr(self.ml_page, "info"):
            self.ml_page.info.connect(self.on_info)

        # 日志页与 logger 连接（若存在 attach 接口）
        if self.log_page:
            if hasattr(self.log_page, "attach_python_logging"):
                self.log_page.attach_python_logging(self._log)

        # —— 组装中心区 ——
        central = QWidget()
        lay = QVBoxLayout()
        lay.addWidget(self.header)
        lay.addWidget(self.tabs, 1)
        central.setLayout(lay)
        self.setCentralWidget(central)

        # —— 加入标签（保持原顺序，在“外设控制”后插入“模型训练与优化”，最后“系统日志”）——
        if self.dashboard_page:
            self.tabs.addTab(self.dashboard_page, "主界面")
        self.tabs.addTab(self.task_page, "运动范式")
        self.tabs.addTab(self.eeg_page, "脑电分类")
        self.tabs.addTab(self.device_page, "外设控制")
        if self.ml_page:
            self.tabs.addTab(self.ml_page, "模型训练与优化")
        if self.log_page:
            self.tabs.addTab(self.log_page, "系统日志")

        # —— 状态栏 ——
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("系统就绪：先在“脑电分类”连接/训练（或演示）→ 在“运动范式”开始试次 → 如需自动给外设下发指令，在“外设控制”勾选自动发送。")
        self.setStyleSheet("QMainWindow { background: #FFFFFF; }")

    # ================== 槽函数：跨模块联动 ==================
    def on_stage(self, name: str, idx: int):
        """范式阶段变化：驱动 EEG trial 开始/结束"""
        self.status.showMessage(f"当前环节：{name}")
        # 依据范式选择判断左右
        current_is_left = True
        if hasattr(self.task_page, "task"):
            try:
                current_is_left = (self.task_page.task.currentIndex() == 0)
            except Exception:
                current_is_left = True

        if name == "运动想象":
            # 开始试次投票窗口
            if hasattr(self.eeg_page, "begin_trial"):
                self.eeg_page.begin_trial('left' if current_is_left else 'right')

        if name == "休息结束":
            # 结束投票并产出结果
            if hasattr(self.eeg_page, "end_trial"):
                self.eeg_page.end_trial('left' if current_is_left else 'right')

    def on_info(self, text: str):
        self.status.showMessage(text)
        # 同步写入 logger（若有日志页，会被捕捉显示）
        try:
            self._log.info(text)
        except Exception:
            pass

    def on_progress(self, done: int, total: int):
        self.status.showMessage(f"循环进度：{done}/{total}")

    def on_trial_result(self, pred: str, success: bool):
        """EEG 产出一次试次结果 → 回灌到范式统计 → 外设页处理"""
        intended = "左手抓握"
        try:
            intended = "左手抓握" if self.task_page.task.currentIndex() == 0 else "右手抓握"
        except Exception:
            pass

        # 回灌范式统计
        if hasattr(self.task_page, "notify_trial_result"):
            self.task_page.notify_trial_result(pred, success, intended)

        self.status.showMessage(f"试次结果：预测={pred} 成功={success}")

        # 传给外设页（若其实现自动发送，将触发发送）
        if hasattr(self.device_page, "handle_trial_result"):
            try:
                self.device_page.handle_trial_result(pred, success)
            except Exception:
                pass

    def on_device_send_result(self, ok: bool, message: str):
        """外设发送成功/失败 → 回灌到范式统计"""
        if hasattr(self.task_page, "notify_device_send"):
            try:
                self.task_page.notify_device_send(ok, message)
            except Exception:
                pass
        tip = "成功" if ok else "失败"
        self.status.showMessage(f"外设发送{tip}：{message}")

    def on_device_feedback(self, feedback: str):
        """外设即时反馈（如 ACK/ERR/状态）"""
        self.status.showMessage(f"外设反馈：{feedback}")
        try:
            self._log.info("设备反馈: %s", feedback)
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    dlg = LoginDialog()
    if dlg.exec_() != QDialog.Accepted:
        return
    username = dlg.username_edit.text().strip() or "未命名用户"
    w = MainWindow(username)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
