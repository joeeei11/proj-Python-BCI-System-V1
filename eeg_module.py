# -*- coding: utf-8 -*-
# eeg_module.py
#
# 脑电波数据处理与分类模块（可独立运行 / 可嵌入主程序）
# 功能：采集（串口/蓝牙/演示）-> 预处理(陷波/带通) -> CSP特征 -> SVM/KNN分类 -> 实时反馈
# 与范式联动：begin_trial(label) 开始一次投票；end_trial(intended) 输出试次结果（成功/失败）
#
# 使用说明（零基础也能懂）：
# 1）没有设备？选择“演示模式”，点“连接”，再点“在线分类”，即可看到模型/投票结果（随机但可控的演示数据）。
# 2）有设备（串口）？选择“串口”，填串口号（如 COM5）和波特率（默认115200），点“连接”即可。
# 3）蓝牙？需要安装 pybluez，选择“蓝牙”，填地址（如 00:11:22:33:44:55），点“连接”。
# 4）快速校准：点“采集左手样本 N段”“采集右手样本 N段”，然后“训练模型”，模型就能用于在线分类。
# 5）与范式页连用：范式进入“运动想象”时调用 begin_trial('left'/'right')；结束时 end_trial('left'/'right') 得到结果。

import sys
import os
import time
import threading
import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox, QMessageBox, QCheckBox
)

# 第三方科学计算/机器学习
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 可选：串口、蓝牙
try:
    import serial
except ImportError:
    serial = None

try:
    from bluetooth import BluetoothSocket, RFCOMM   # PyBluez
except Exception:
    BluetoothSocket = None
    RFCOMM = None


# --------------------------- 工具：环形缓冲区 ---------------------------
class RingBuffer:
    """简单环形缓冲区，用于保存最近的 EEG 数据（按通道存）"""
    def __init__(self, n_channels, maxlen):
        self.n_channels = n_channels
        self.maxlen = maxlen
        self.buf = np.zeros((maxlen, n_channels), dtype=np.float32)
        self.idx = 0
        self.full = False

    def append(self, samples):
        """samples: (n, n_channels)"""
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        n = samples.shape[0]
        for i in range(n):
            self.buf[self.idx, :] = samples[i]
            self.idx += 1
            if self.idx >= self.maxlen:
                self.idx = 0
                self.full = True

    def get_last(self, n):
        """取最近 n 个点"""
        if not self.full and self.idx < n:
            return None
        end = self.idx
        start = (end - n) % self.maxlen
        if start < end:
            return self.buf[start:end, :].copy()
        else:
            return np.vstack((self.buf[start:, :], self.buf[:end, :])).copy()


# --------------------------- 工具：CSP 变换器 ---------------------------
class CSP:
    """
    简化版 CSP（用于二分类：左/右）
    - fit(X, y): X shape=(n_trials, n_channels, n_samples), y ∈ {0,1}
    - transform(X): 返回 log-variance 特征
    选取每类 m 个滤波器（默认 m=2），总特征维度为 2m
    """
    def __init__(self, m=2):
        self.m = m
        self.W = None  # 空间滤波矩阵
        self.select_idx = None  # 选取的列索引

    @staticmethod
    def _cov(x):
        # x: (n_channels, n_samples)
        x = x - x.mean(axis=1, keepdims=True)
        c = np.dot(x, x.T) / (x.shape[1] - 1)
        return c / np.trace(c)

    def fit(self, X, y):
        c1 = np.zeros((X.shape[1], X.shape[1]))
        c2 = np.zeros_like(c1)
        n1 = n2 = 0
        for i in range(X.shape[0]):
            c = self._cov(X[i])
            if y[i] == 0:
                c1 += c; n1 += 1
            else:
                c2 += c; n2 += 1
        c1 /= max(1, n1)
        c2 /= max(1, n2)
        # 广义特征分解：C1 v = λ (C1 + C2) v
        C = c1 + c2
        # 防奇异
        C += 1e-6 * np.eye(C.shape[0])
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(C).dot(c1))
        # 按特征值排序（从大到小）
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        # 选择前 m 和后 m，共 2m 个
        m = self.m
        sel = np.hstack([np.arange(m), np.arange(-m, 0)])
        self.W = eigvecs
        self.select_idx = sel

    def transform(self, X):
        # X shape=(n_trials, n_channels, n_samples)
        if self.W is None or self.select_idx is None:
            raise RuntimeError("CSP 未训练，请先调用 fit()")
        feats = []
        for i in range(X.shape[0]):
            Z = self.W.T.dot(X[i])  # 空间滤波
            Z = Z[self.select_idx, :]
            var = np.var(Z, axis=1)
            feats.append(np.log(var / np.sum(var)))
        return np.array(feats)


# --------------------------- 采集线程（串口/蓝牙/演示） ---------------------------
class AcqThread(QThread):
    """
    采集线程：
    - mode: 'demo'/'serial'/'bt'
    - emit samples_ready(np.ndarray shape=(n, n_channels))
    """
    samples_ready = pyqtSignal(object)
    connected = pyqtSignal(bool, str)

    def __init__(self, mode='demo', serial_port='COM5', baud=115200, bt_addr='', srate=250, n_channels=8, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.serial_port = serial_port
        self.baud = baud
        self.bt_addr = bt_addr
        self.srate = srate
        self.n_channels = n_channels
        self._running = False
        self._ser = None
        self._bt = None
        # 演示信号参数
        self._t = 0.0
        self._dt = 1.0 / self.srate
        self._demo_label = 0  # 0=left, 1=right（用于演示时改变谱特征）

    def set_demo_label(self, label01):
        self._demo_label = int(label01)

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        ok = False
        msg = ""

        try:
            if self.mode == 'serial':
                if serial is None:
                    raise RuntimeError("未安装 pyserial，无法使用串口")
                self._ser = serial.Serial(self.serial_port, self.baud, timeout=0.2)
                ok = True; msg = f"串口已连接：{self.serial_port}@{self.baud}"
            elif self.mode == 'bt':
                if BluetoothSocket is None:
                    raise RuntimeError("未安装 pybluez，无法使用蓝牙")
                self._bt = BluetoothSocket(RFCOMM)
                self._bt.connect((self.bt_addr, 1))
                ok = True; msg = f"蓝牙已连接：{self.bt_addr}"
            else:
                ok = True; msg = "演示模式已启动"

        except Exception as e:
            ok = False; msg = f"连接失败：{e}"

        self.connected.emit(ok, msg)
        if not ok:
            return

        # 主循环：每 40ms 打包一次
        pack = max(1, int(self.srate * 0.04))  # ~25Hz推送
        while self._running:
            try:
                if self.mode == 'serial':
                    # 假设设备每行输出 n_channels 个浮点/整型，用逗号分隔
                    # 按 pack 行读取
                    rows = []
                    while len(rows) < pack and self._running:
                        line = self._ser.readline().decode(errors='ignore').strip()
                        if not line:
                            break
                        parts = line.replace(';', ',').split(',')
                        if len(parts) >= self.n_channels:
                            vals = [float(x) for x in parts[:self.n_channels]]
                            rows.append(vals)
                    if rows:
                        arr = np.array(rows, dtype=np.float32)
                        self.samples_ready.emit(arr)

                elif self.mode == 'bt':
                    # 按照设备协议拆包（这里示例为每帧 n_channels*4 字节的float）
                    # 实际需替换为你的蓝牙设备协议
                    data = self._bt.recv(self.n_channels * 4 * pack)
                    if data:
                        arr = np.frombuffer(data, dtype=np.float32)
                        arr = arr.reshape(-1, self.n_channels)
                        self.samples_ready.emit(arr)

                else:  # demo：生成合成EEG
                    # 8-30Hz带内噪声 + 左右手差异（比如 10Hz/12Hz 两个分量不同幅度）
                    n = pack
                    t = self._t + np.arange(n) * self._dt
                    sig = np.random.randn(n, self.n_channels) * 5e-6  # 噪声
                    f1, f2 = (10.0, 18.0) if self._demo_label == 0 else (12.0, 24.0)
                    for ch in range(self.n_channels):
                        sig[:, ch] += 2e-5 * np.sin(2*np.pi*f1*t + ch*0.2)
                        sig[:, ch] += 1e-5 * np.sin(2*np.pi*f2*t + ch*0.1)
                    self._t = t[-1]
                    self.samples_ready.emit(sig.astype(np.float32))
                    self.msleep(40)

            except Exception as e:
                self.connected.emit(False, f"采集中断：{e}")
                break

        # 退出清理
        try:
            if self._ser:
                self._ser.close()
            if self._bt:
                self._bt.close()
        except:
            pass
        self.connected.emit(False, "连接已关闭")


# --------------------------- EEG 模块主界面 ---------------------------
class EEGModule(QWidget):
    """
    EEG 采集 + 预处理 + CSP + SVM/KNN 在线分类
    对外信号：
      - info(str): 文本提示
      - classified(str, float): 分类结果标签 与 置信度/评分
      - trial_result(str, bool): 本次试次预测标签、是否成功（与期望一致）
    对外方法（供主程序/范式页调用）：
      - begin_trial(label_str): 开始投票窗口（运动想象期开始）
      - end_trial(intended_label_str): 结束投票并输出 trial_result
    """
    info = pyqtSignal(str)
    classified = pyqtSignal(str, float)
    trial_result = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("脑电数据与分类")

        # 采样与通道设置（根据设备调整）
        self.srate = 250
        self.n_channels = 8

        # 预处理参数
        self.band_lo = 8.0
        self.band_hi = 30.0
        self.notch = 50.0  # 工频（中国 50Hz）

        # 缓冲/窗口/步长
        self.window_sec = 1.0
        self.step_sec = 0.5

        # 运行状态
        self._acq = None
        self._buffer = RingBuffer(self.n_channels, int(self.srate * 10))  # 10秒缓冲
        self._online_timer = QTimer(self)
        self._online_timer.timeout.connect(self._on_online_tick)
        self._online_enable = False

        # 训练与模型
        self._calib_left = []   # list of (n_channels, n_samples)
        self._calib_right = []
        self.csp = CSP(m=2)
        self.scaler = StandardScaler()
        self.clf_name = "SVM"    # 或 "KNN"
        self.clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=0)

        # 投票（配合范式）
        self._voting = False
        self._votes = []   # 收集一次运动想象期内的预测标签
        self._last_pred = None

        self._build_ui()
        self._apply_styles()

    # ---------------- UI 搭建 ----------------
    def _build_ui(self):
        font = QFont("Microsoft YaHei", 11, QFont.Bold)
        self.setFont(font)

        # 连接参数
        conn_group = QGroupBox("连接")
        g = QGridLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["演示模式", "串口", "蓝牙"])
        # 没有蓝牙库就禁用
        if BluetoothSocket is None:
            self.mode_combo.model().item(2).setEnabled(False)

        self.port_edit = QLineEdit("COM5")   # 串口号
        self.baud_edit = QLineEdit("115200")
        self.bt_edit = QLineEdit("00:11:22:33:44:55")
        self.btn_connect = QPushButton("连接")
        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setEnabled(False)

        g.addWidget(QLabel("模式"), 0, 0);      g.addWidget(self.mode_combo, 0, 1)
        g.addWidget(QLabel("串口号"), 1, 0);    g.addWidget(self.port_edit, 1, 1)
        g.addWidget(QLabel("波特率"), 2, 0);    g.addWidget(self.baud_edit, 2, 1)
        g.addWidget(QLabel("蓝牙地址"), 3, 0);  g.addWidget(self.bt_edit, 3, 1)
        g.addWidget(self.btn_connect, 4, 0);   g.addWidget(self.btn_disconnect, 4, 1)
        conn_group.setLayout(g)

        # 在线分类控制
        online_group = QGroupBox("在线分类")
        h = QHBoxLayout()
        self.chk_online = QCheckBox("在线分类（每0.5秒滑窗）")
        self.chk_online.toggled.connect(self._toggle_online)
        self.res_label = QLabel("结果：——")
        h.addWidget(self.chk_online); h.addStretch(1); h.addWidget(self.res_label)
        online_group.setLayout(h)

        # 校准与训练
        calib_group = QGroupBox("快速校准（CSP + 分类器）")
        cg = QGridLayout()
        self.win_spin = QDoubleSpinBox(); self.win_spin.setRange(0.5, 3.0); self.win_spin.setSingleStep(0.5); self.win_spin.setValue(1.0)
        self.nseg_spin = QSpinBox(); self.nseg_spin.setRange(2, 50); self.nseg_spin.setValue(10)
        self.btn_cap_left = QPushButton("采集左手样本")
        self.btn_cap_right = QPushButton("采集右手样本")
        self.btn_train = QPushButton("训练模型")
        self.clf_combo = QComboBox(); self.clf_combo.addItems(["SVM", "KNN"])
        self.clf_combo.currentTextChanged.connect(self._on_clf_change)

        cg.addWidget(QLabel("单段时长(秒)"), 0, 0); cg.addWidget(self.win_spin, 0, 1)
        cg.addWidget(QLabel("段数"), 1, 0);       cg.addWidget(self.nseg_spin, 1, 1)
        cg.addWidget(self.btn_cap_left, 2, 0);   cg.addWidget(self.btn_cap_right, 2, 1)
        cg.addWidget(QLabel("分类器"), 3, 0);     cg.addWidget(self.clf_combo, 3, 1)
        cg.addWidget(self.btn_train, 4, 0, 1, 2)
        calib_group.setLayout(cg)

        # 底部按钮事件
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_cap_left.clicked.connect(lambda: self._capture_samples(0))
        self.btn_cap_right.clicked.connect(lambda: self._capture_samples(1))
        self.btn_train.clicked.connect(self._train_model)

        # 总布局
        root = QVBoxLayout()
        root.addWidget(conn_group)
        root.addWidget(online_group)
        root.addWidget(calib_group)
        root.addStretch(1)
        self.setLayout(root)

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background: #FFFFFF;
                color: #323232;
                font-family: "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #E6E6E6;
                border-radius: 12px;
                padding: 12px;
                margin-top: 8px;
                background: #FAFAFA;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                padding: 6px 10px;
                background: #F7F7F7;
                min-width: 140px;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border: 1px solid #007AFF;
                background: #FFFFFF;
            }
            QPushButton {
                background: #007AFF;
                color: white;
                padding: 10px 18px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover { background: #1A84FF; }
            QPushButton:pressed { background: #0062CC; }
            QPushButton:disabled { background: #E0E0E0; color: #9E9E9E; }
        """)

    # ---------------- 连接/断开 ----------------
    def _connect(self):
        mode_idx = self.mode_combo.currentIndex()
        if mode_idx == 0:
            mode = 'demo'
            port = ''
            baud = 0
            bt = ''
        elif mode_idx == 1:
            mode = 'serial'
            port = self.port_edit.text().strip()
            baud = int(self.baud_edit.text().strip() or 115200)
            bt = ''
            if serial is None:
                QMessageBox.critical(self, "错误", "未安装 pyserial，不能使用串口。请先安装：pip install pyserial")
                return
        else:
            mode = 'bt'
            port = ''
            baud = 0
            bt = self.bt_edit.text().strip()
            if BluetoothSocket is None:
                QMessageBox.critical(self, "错误", "未安装 pybluez，不能使用蓝牙。请先安装：pip install pybluez")
                return

        # 启动采集线程
        self._buffer = RingBuffer(self.n_channels, int(self.srate * 10))
        self._acq = AcqThread(mode=mode, serial_port=port, baud=baud, bt_addr=bt,
                              srate=self.srate, n_channels=self.n_channels)
        self._acq.samples_ready.connect(self._on_samples)
        self._acq.connected.connect(self._on_connected)
        self._acq.start()

    def _disconnect(self):
        if self._acq:
            self._acq.stop()
            self._acq.wait(1000)
            self._acq = None
        self.info.emit("连接已关闭")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)

    def _on_connected(self, ok, msg):
        self.info.emit(msg)
        if ok:
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
        else:
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)

    # ---------------- 数据接收/预处理/在线分类 ----------------
    def _on_samples(self, arr):
        """接收采集线程推送的数据，进入环形缓冲区"""
        self._buffer.append(arr)

    def _design_filters(self):
        """设计陷波与带通滤波器（一次即可）"""
        # 工频陷波（IIR notch）
        w0 = self.notch / (self.srate / 2.0)
        bw = w0 / 35.0  # Q约 35
        self._b_notch, self._a_notch = iirnotch(w0, Q=35)

        # 带通（8-30Hz）
        wp = [self.band_lo / (self.srate/2.0), self.band_hi / (self.srate/2.0)]
        self._b_bp, self._a_bp = butter(4, wp, btype='bandpass')

    def _preprocess(self, x):
        """预处理单段：x shape=(n_samples, n_channels) -> (n_channels, n_samples)"""
        if not hasattr(self, "_b_notch"):
            self._design_filters()
        # 逐通道滤波
        y = np.zeros_like(x, dtype=np.float32)
        for ch in range(x.shape[1]):
            sig = x[:, ch]
            # 陷波去工频
            sig = filtfilt(self._b_notch, self._a_notch, sig)
            # 带通
            sig = filtfilt(self._b_bp, self._a_bp, sig)
            y[:, ch] = sig
        return y.T  # (n_channels, n_samples)

    def _on_online_tick(self):
        """在线分类定时器：每 step_sec 取一次窗口，提取特征并预测"""
        n_win = int(self.window_sec * self.srate)
        data = self._buffer.get_last(n_win)
        if data is None:
            return
        seg = self._preprocess(data)  # (n_channels, n_samples)

        # 如果还没训练，给出提示
        if self.csp.W is None or self.clf is None:
            self.info.emit("模型未训练，无法在线分类。请先‘快速校准->训练模型’。")
            return

        feats = self.csp.transform(seg[np.newaxis, :, :])  # (1, feat_dim)
        feats = self.scaler.transform(feats)
        if self.clf_name == "SVM":
            prob = self.clf.predict_proba(feats)[0]
        else:
            prob = self.clf.predict_proba(feats)[0]
        idx = int(np.argmax(prob))
        label = "left" if idx == 0 else "right"
        score = float(np.max(prob))
        self.res_label.setText(f"结果：{label}  置信度={score:.2f}")
        self.classified.emit(label, score)

        # 若在范式投票窗口内，记录票
        if self._voting:
            self._votes.append(label)

    def _toggle_online(self, on):
        self._online_enable = on
        if on:
            self._online_timer.start(int(self.step_sec * 1000))
            self.info.emit("在线分类已开启")
        else:
            self._online_timer.stop()
            self.info.emit("在线分类已关闭")

    # ---------------- 快速校准：采集样本 & 训练 ----------------
    def _capture_samples(self, label01):
        """从当前缓冲取固定时长，重复 n 段，作为校准样本。演示/真机都可用。"""
        nseg = self.nseg_spin.value()
        win = self.win_spin.value()
        n = int(win * self.srate)

        if self._acq is None:
            QMessageBox.information(self, "提示", "请先‘连接’再采集样本。")
            return

        dest = self._calib_left if label01 == 0 else self._calib_right
        name = "左手" if label01 == 0 else "右手"

        taken = 0
        while taken < nseg:
            data = self._buffer.get_last(n)
            if data is not None:
                seg = self._preprocess(data)  # (n_channels, n_samples)
                dest.append(seg)
                taken += 1
                self.info.emit(f"已采集{name}样本 {taken}/{nseg}")
                QApplication.processEvents()
                time.sleep(0.1)
            else:
                time.sleep(0.05)

        QMessageBox.information(self, "完成", f"{name}样本采集完成：{nseg} 段。")

    def _on_clf_change(self, text):
        self.clf_name = text
        if text == "SVM":
            self.clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=0)
        else:
            self.clf = KNeighborsClassifier(n_neighbors=5)

    def _train_model(self):
        if len(self._calib_left) < 2 or len(self._calib_right) < 2:
            QMessageBox.warning(self, "样本不足", "左右手样本都至少需要 2 段。")
            return
        X = np.array(self._calib_left + self._calib_right)  # (n_trials, n_channels, n_samples)
        y = np.array([0]*len(self._calib_left) + [1]*len(self._calib_right))
        # CSP
        self.csp.fit(X, y)
        feats = self.csp.transform(X)
        # 归一化+训练
        self.scaler.fit(feats)
        feats = self.scaler.transform(feats)
        self.clf.fit(feats, y)
        QMessageBox.information(self, "完成", f"模型训练完成（分类器：{self.clf_name}）。现在可开启‘在线分类’或与范式联动。")
        self.info.emit("模型训练完成")

    # ---------------- 与范式联动：投票接口 ----------------
    def begin_trial(self, label_str: str):
        """
        在范式模块进入“运动想象期”时调用：
        - label_str: 'left' 或 'right'（仅用于演示模式生成对应特征，真机不影响）
        - 开始清空投票，若演示模式则通知采集线程改变模拟信号模式
        """
        self._votes = []
        self._voting = True
        if self._acq and isinstance(self._acq, AcqThread) and self._acq.mode == 'demo':
            self._acq.set_demo_label(0 if label_str == 'left' else 1)
        self.info.emit(f"开始试次投票窗口（期望：{label_str}）")

    def end_trial(self, intended_label: str):
        """
        范式“休息期”时调用：结束投票，统计多数票作为本次预测
        返回 trial_result 信号（预测标签, 是否成功）
        """
        self._voting = False
        if not self._votes:
            # 没有票，按当前在线结果或未知
            pred = self._last_pred or "unknown"
        else:
            pred = "left" if self._votes.count("left") >= self._votes.count("right") else "right"
        success = (pred == intended_label)
        self.trial_result.emit(pred, success)
        self.info.emit(f"试次结束：预测={pred}，目标={intended_label}，成功={success}")
        return pred, success


# 独立运行（调试用）
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = EEGModule()
    w.show()
    sys.exit(app.exec_())
