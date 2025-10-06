# -*- coding: utf-8 -*-
# data_module.py
#
# 数据存储与分析模块（作为一个标签页加入主窗口）
# - 训练日志：SQLite 持久化
# - 图表分析：学习曲线（按会话/周/月）、Welch t 检验、混淆矩阵
# - EEG CSV 回放：多通道叠加、Butterworth 滤波、功率谱热力图（通道×频率）
# - 接口：notify_trial_started / notify_trial_result / notify_device_send
#
# 稳定性处理：
# - Figure 使用 constrained_layout=True（替代 tight_layout）
# - 所有绘图刷新使用 draw_idle()（替代 draw()）
# - 增加可见性判断与 50ms 去抖，避免频繁布局重排造成崩溃
#
# CSV 固定格式约定：
#   time,C3,Cz,C4,CP3,CPz,CP4,...
#   time 单位：秒（浮点），其余列为微伏值（浮点）

import os
import sqlite3
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox
)

# ---- Matplotlib 画布（统一使用 constrained_layout，避免 tight_layout 导致的崩溃）----
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

APPLE_BLUE = "#007AFF"
BORDER = "#E6E6E6"
TEXT = "#323232"
YAHEI = QFont("Microsoft YaHei", 10, QFont.Bold)

# ============ 小工具函数 ============

def butter_filter(data, fs, f_low=None, f_high=None, btype='band', order=4):
    """Butterworth 滤波封装。
    data: ndarray (n_samples,) 或 (n_samples, n_channels)
    fs: 采样率
    f_low/f_high: 截止频率
    btype: 'band'/'low'/'high'
    """
    if fs <= 0 or data is None:
        return data
    nyq = fs * 0.5
    if btype == 'band':
        if not f_low or not f_high or f_low <= 0 or f_high >= nyq or f_low >= f_high:
            return data
        Wn = [f_low / nyq, f_high / nyq]
    elif btype == 'low':
        if not f_high or f_high <= 0 or f_high >= nyq:
            return data
        Wn = f_high / nyq
    elif btype == 'high':
        if not f_low or f_low <= 0 or f_low >= nyq:
            return data
        Wn = f_low / nyq
    else:
        return data
    try:
        b, a = signal.butter(order, Wn, btype=btype)
        return signal.filtfilt(b, a, data, axis=0)
    except Exception:
        return data

def compute_welch_psd(x, fs, nperseg=512, noverlap=256):
    """计算 Welch PSD（功率谱密度）。返回 f, pxx。"""
    try:
        f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
        return f, pxx
    except Exception:
        # 兜底：返回空
        return np.array([]), np.array([])

# ============ Matplotlib 画布封装（避免 tight_layout） ============

class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=5, dpi=120):
        # 使用 constrained_layout 避免 tight_layout 在复杂场景导致的崩溃
        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(fig)
        self.fig = fig

# ============ 主类 ============

class DataAnalyticsPanel(QWidget):
    """数据存储与分析页"""
    info = pyqtSignal(str)

    def __init__(self, db_path="data.db", parent=None):
        super().__init__(parent)
        self.setFont(YAHEI)
        self.setStyleSheet(f"""
            QWidget {{ background:#FFFFFF; color:{TEXT}; font-family:"Microsoft YaHei","微软雅黑",Arial; font-size:14px; }}
            QGroupBox {{ border:1px solid {BORDER}; border-radius:12px; padding:10px; margin-top:8px; background:#FAFAFA; font-weight:bold; }}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding:0 6px; }}
            QPushButton {{ background:{APPLE_BLUE}; color:#FFF; padding:8px 14px; border-radius:8px; font-weight:bold; border:none; }}
            QPushButton:hover {{ background:#1A84FF; }} QPushButton:pressed {{ background:#0062CC; }}
            QPushButton:disabled {{ background:#E0E0E0; color:#9E9E9E; }}
            QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox {{ background:#FFF; border:1px solid #D0D0D0; border-radius:8px; padding:4px 8px; }}
            QTableWidget {{ background:#FFF; border:1px solid #E0E0E0; border-radius:8px; }}
        """)

        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

        # 待写入的一次试次（等待外设发送结果后入库）
        self._pending_trial = {}

        # EEG CSV（pandas DataFrame）
        self._eeg_df = None
        self._fs_hint = 250.0

        # 画布与去抖
        self._redraw_pending = False

        self._build_ui()
        self.refresh_table()
        self._draw_all()

    # ---------- DB 结构 ----------
    def _create_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                session TEXT,
                username TEXT,
                intended_label TEXT,   -- "左手抓握"/"右手抓握"
                predicted TEXT,        -- "left"/"right"/NULL
                success INTEGER,       -- 0/1/NULL
                send_ok INTEGER,       -- 0/1/NULL
                message TEXT,
                fix_s REAL, cue_s REAL, imag_s REAL, rest_s REAL
            )
        """)
        self.conn.commit()

    # ---------- UI ----------
    def _build_ui(self):
        # 顶部操作条
        top_box = QGroupBox("训练日志与导出")
        tl = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新日志")
        self.btn_export_csv = QPushButton("导出 CSV")
        self.btn_export_json= QPushButton("导出 JSON")
        self.btn_refresh.clicked.connect(self.refresh_table)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)
        tl.addWidget(self.btn_refresh); tl.addWidget(self.btn_export_csv); tl.addWidget(self.btn_export_json)
        tl.addStretch(1)
        top_box.setLayout(tl)

        # 日志表
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "时间", "会话", "用户", "意图", "预测", "成功", "外设发送", "信息",
            "注视(s)", "提示(s)", "想象(s)"
        ] + ["休息(s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)

        # 统计参数
        stat_box = QGroupBox("统计与图表")
        sg = QGridLayout()
        self.cmb_curve = QComboBox(); self.cmb_curve.addItems(["按会话（日）","按周","按月"])
        self.btn_draw = QPushButton("刷新图表")
        self.btn_draw.clicked.connect(self._draw_all)
        # t 检验说明
        self.lab_ttest = QLabel("Welch t 检验：比较左右意图成功率差异（p 值）")
        sg.addWidget(QLabel("学习曲线聚合"), 0, 0); sg.addWidget(self.cmb_curve, 0, 1)
        sg.addWidget(self.btn_draw, 0, 2); sg.addWidget(self.lab_ttest, 0, 3, 1, 2)
        stat_box.setLayout(sg)

        # EEG 回放设置
        eeg_box = QGroupBox("EEG CSV 回放与功率热力图")
        eg = QGridLayout()
        self.btn_load_csv = QPushButton("加载 EEG CSV")
        self.btn_load_csv.clicked.connect(self._load_eeg_csv)

        self.spin_fs  = QDoubleSpinBox(); self.spin_fs.setRange(1, 5000); self.spin_fs.setDecimals(1); self.spin_fs.setValue(self._fs_hint)
        self.cmb_filter = QComboBox(); self.cmb_filter.addItems(["不滤波","带通(8-30)","低通","高通","自定义带通"])
        self.spin_f1 = QDoubleSpinBox(); self.spin_f1.setRange(0.1, 1000); self.spin_f1.setDecimals(1); self.spin_f1.setValue(8.0)
        self.spin_f2 = QDoubleSpinBox(); self.spin_f2.setRange(0.1, 1000); self.spin_f2.setDecimals(1); self.spin_f2.setValue(30.0)
        self.spin_down = QSpinBox(); self.spin_down.setRange(1, 50); self.spin_down.setValue(4)  # 下采样因子
        self.btn_redraw_eeg = QPushButton("刷新EEG图")
        self.btn_redraw_eeg.clicked.connect(self._draw_all)

        eg.addWidget(self.btn_load_csv, 0,0)
        eg.addWidget(QLabel("采样率(Hz)"), 0,1); eg.addWidget(self.spin_fs, 0,2)
        eg.addWidget(QLabel("滤波"), 1,0); eg.addWidget(self.cmb_filter, 1,1)
        eg.addWidget(QLabel("f1/低通(Hz)"), 1,2); eg.addWidget(self.spin_f1, 1,3)
        eg.addWidget(QLabel("f2/高通(Hz)"), 1,4); eg.addWidget(self.spin_f2, 1,5)
        eg.addWidget(QLabel("下采样"), 1,6); eg.addWidget(self.spin_down, 1,7)
        eg.addWidget(self.btn_redraw_eeg, 1,8)
        eeg_box.setLayout(eg)

        # 画布：四象限布局（学习曲线、混淆矩阵、EEG叠加、功率热力图）
        self.canvas = MplCanvas(width=10, height=7, dpi=120)
        self.ax_curve = self.canvas.fig.add_subplot(2,2,1)   # 学习曲线
        self.ax_cm    = self.canvas.fig.add_subplot(2,2,2)   # 混淆矩阵
        self.ax_eeg   = self.canvas.fig.add_subplot(2,2,3)   # 叠加波形
        self.ax_spec  = self.canvas.fig.add_subplot(2,2,4)   # 频谱热力

        # 布局
        root = QVBoxLayout()
        root.addWidget(top_box)
        root.addWidget(self.table, 2)
        root.addWidget(stat_box)
        root.addWidget(eeg_box)
        root.addWidget(self.canvas, 4)
        self.setLayout(root)

    # ---------- 表格刷新/导出 ----------
    def refresh_table(self):
        df = self._read_df()
        self._fill_table(df)
        self.info.emit(f"已刷新日志，共 {len(df)} 条记录")
        # 刷新图表（去抖）
        self._debounced_draw()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "trials.csv", "CSV Files (*.csv)")
        if not path: return
        df = self._read_df()
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self.info.emit(f"导出 CSV 成功：{path}")
        except Exception as e:
            self.info.emit(f"导出 CSV 失败：{e}")

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", "trials.json", "JSON Files (*.json)")
        if not path: return
        df = self._read_df()
        try:
            df.to_json(path, orient="records", force_ascii=False, indent=2)
            self.info.emit(f"导出 JSON 成功：{path}")
        except Exception as e:
            self.info.emit(f"导出 JSON 失败：{e}")

    def _read_df(self):
        try:
            df = pd.read_sql_query("SELECT * FROM trials ORDER BY id DESC", self.conn)
        except Exception:
            df = pd.DataFrame(columns=[
                "ts","session","username","intended_label","predicted","success",
                "send_ok","message","fix_s","cue_s","imag_s","rest_s"
            ])
        return df

    def _fill_table(self, df: pd.DataFrame):
        self.table.setRowCount(0)
        if df is None or df.empty:
            return
        cols = ["ts","session","username","intended_label","predicted","success","send_ok","message","fix_s","cue_s","imag_s","rest_s"]
        for _, row in df.iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, key in enumerate(cols):
                val = row.get(key, "")
                if key in ("success","send_ok"):
                    val = "是" if int(val)==1 else ("否" if pd.notna(val) else "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)

    # ---------- 回灌接口 ----------
    def notify_trial_started(self, username: str, intended_label: str,
                              fix_s: float, cue_s: float, imag_s: float, rest_s: float):
        """在“注视点”阶段由主程序调用，缓存一次试次，等待预测与外设发送后落库"""
        self._pending_trial = dict(
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session=datetime.now().strftime("%Y-%m-%d"),
            username=username,
            intended_label=intended_label,
            predicted=None, success=None, send_ok=None, message=None,
            fix_s=float(fix_s), cue_s=float(cue_s), imag_s=float(imag_s), rest_s=float(rest_s)
        )
        self.info.emit(f"记录准备：{intended_label}（{username}）")

    def notify_trial_result(self, predicted: str, success: bool):
        """EEG 模块给出预测后，由主程序调用"""
        if not self._pending_trial:
            # 没有 pending，创建临时记录（等待发送结果入库）
            self._pending_trial = dict(
                ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session=datetime.now().strftime("%Y-%m-%d"),
                username="未命名用户", intended_label="未知",
                predicted=predicted, success=bool(success),
                send_ok=None, message=None, fix_s=None, cue_s=None, imag_s=None, rest_s=None
            )
        else:
            self._pending_trial["predicted"] = predicted
            self._pending_trial["success"] = bool(success)

    def notify_device_send(self, send_ok: bool, message: str):
        """外设发送完成后，由主程序调用；此时真正入库并刷新表格与图表"""
        # 若还没有 pending，也可以单独落库
        if not self._pending_trial:
            self._pending_trial = dict(
                ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session=datetime.now().strftime("%Y-%m-%d"),
                username="未命名用户", intended_label="未知",
                predicted=None, success=None,
                fix_s=None, cue_s=None, imag_s=None, rest_s=None
            )
        self._pending_trial["send_ok"] = bool(send_ok)
        self._pending_trial["message"] = str(message)

        try:
            c = self.conn.cursor()
            c.execute("""
                INSERT INTO trials
                (ts, session, username, intended_label, predicted, success, send_ok, message,
                 fix_s, cue_s, imag_s, rest_s)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                self._pending_trial.get("ts"),
                self._pending_trial.get("session"),
                self._pending_trial.get("username"),
                self._pending_trial.get("intended_label"),
                self._pending_trial.get("predicted"),
                int(self._pending_trial["success"]) if self._pending_trial.get("success") is not None else None,
                int(self._pending_trial["send_ok"]) if self._pending_trial.get("send_ok") is not None else None,
                self._pending_trial.get("message"),
                self._pending_trial.get("fix_s"),
                self._pending_trial.get("cue_s"),
                self._pending_trial.get("imag_s"),
                self._pending_trial.get("rest_s"),
            ))
            self.conn.commit()
            self.info.emit("训练记录已保存")
        except Exception as e:
            self.info.emit(f"保存失败：{e}")
        finally:
            self._pending_trial = {}
            self.refresh_table()  # 刷表 & 去抖绘图

    # ---------- 图表绘制 ----------
    def _debounced_draw(self):
        if self._redraw_pending:
            return
        self._redraw_pending = True
        QTimer.singleShot(50, self._draw_all)

    def _draw_all(self):
        # 不可见或未初始化时直接返回（避免隐藏状态下多次重排）
        if not self.isVisible() or not hasattr(self, "canvas"):
            self._redraw_pending = False
            return

        df = self._read_df()
        # 清理子图
        self.ax_curve.clear(); self.ax_cm.clear(); self.ax_eeg.clear(); self.ax_spec.clear()

        # 1) 学习曲线（按日/周/月聚合的成功率）
        if df is not None and not df.empty:
            try:
                df_ok = df.copy()
                # success 统一到 {0,1}
                if "success" in df_ok.columns:
                    df_ok["success"] = df_ok["success"].apply(lambda x: np.nan if pd.isna(x) else int(x))
                # 时间列
                df_ok["ts"] = pd.to_datetime(df_ok["ts"], errors="coerce")
                df_ok["session"] = pd.to_datetime(df_ok["session"], errors="coerce").dt.date

                agg_mode = self.cmb_curve.currentIndex()  # 0日 1周 2月
                if agg_mode == 0:
                    grp = df_ok.groupby(df_ok["session"])
                elif agg_mode == 1:
                    grp = df_ok.groupby(df_ok["ts"].dt.to_period("W").apply(lambda p: p.start_time.date()))
                else:
                    grp = df_ok.groupby(df_ok["ts"].dt.to_period("M").apply(lambda p: p.start_time.date()))

                curve_x, curve_y = [], []
                for k, g in grp:
                    s = g["success"].dropna()
                    if len(s) == 0:
                        continue
                    curve_x.append(k)
                    curve_y.append(float(np.mean(s)))
                if curve_x:
                    self.ax_curve.plot(curve_x, curve_y, marker="o", color="#007AFF", linewidth=2)
                    self.ax_curve.set_ylim(0, 1)
                    self.ax_curve.set_ylabel("成功率")
                    self.ax_curve.set_title("学习曲线（0~1）")
                    self.ax_curve.grid(True, alpha=0.3)
            except Exception as e:
                self.ax_curve.text(0.5, 0.5, f"学习曲线绘制失败：{e}", ha="center", va="center", transform=self.ax_curve.transAxes)

            # 2) Welch t 检验（左右意图成功率差异）
            try:
                left_s = df_ok[df_ok["intended_label"].astype(str).str.contains("左")]["success"].dropna().astype(int)
                right_s= df_ok[df_ok["intended_label"].astype(str).str.contains("右")]["success"].dropna().astype(int)
                if len(left_s) >= 2 and len(right_s) >= 2:
                    tstat, pval = ttest_ind(left_s, right_s, equal_var=False)
                    self.lab_ttest.setText(f"Welch t 检验：p = {pval:.4f}（n_left={len(left_s)}，n_right={len(right_s)}）")
                else:
                    self.lab_ttest.setText("Welch t 检验：样本不足（左右各≥2条）")
            except Exception as e:
                self.lab_ttest.setText(f"Welch t 检验失败：{e}")

            # 3) 混淆矩阵
            try:
                cm = None
                if "predicted" in df_ok.columns and "intended_label" in df_ok.columns:
                    y_true = df_ok["intended_label"].map(lambda s: "left" if isinstance(s, str) and "左" in s else ("right" if isinstance(s,str) and "右" in s else None))
                    y_pred = df_ok["predicted"]
                    mask = y_true.notna() & y_pred.notna()
                    y_true = y_true[mask]; y_pred = y_pred[mask]
                    if len(y_true) > 0:
                        labels = ["left","right"]
                        cm = confusion_matrix(y_true, y_pred, labels=labels)
                        im = self.ax_cm.imshow(cm, cmap="Blues")
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                self.ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
                        self.ax_cm.set_xticks([0,1]); self.ax_cm.set_xticklabels(["左","右"])
                        self.ax_cm.set_yticks([0,1]); self.ax_cm.set_yticklabels(["左","右"])
                        self.ax_cm.set_xlabel("预测"); self.ax_cm.set_ylabel("真实意图")
                        self.ax_cm.set_title("混淆矩阵")
                        self.ax_cm.figure.colorbar(im, ax=self.ax_cm, fraction=0.046, pad=0.04)
            except Exception as e:
                self.ax_cm.text(0.5, 0.5, f"混淆矩阵失败：{e}", ha="center", va="center", transform=self.ax_cm.transAxes)
        else:
            self.ax_curve.text(0.5, 0.5, "暂无数据", ha="center", va="center", transform=self.ax_curve.transAxes)
            self.lab_ttest.setText("Welch t 检验：暂无数据")
            self.ax_cm.text(0.5, 0.5, "暂无数据", ha="center", va="center", transform=self.ax_cm.transAxes)

        # 4) EEG 叠加与功率热力
        self._draw_eeg_and_spec()

        # 异步刷新并清除去抖标志
        try:
            self.canvas.draw_idle()
        except Exception:
            pass
        self._redraw_pending = False

    # ---------- EEG 绘制 ----------
    def _load_eeg_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 EEG CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if "time" not in df.columns:
                raise ValueError("CSV 必须包含 'time' 列（秒）")
            # 估计采样率
            t = df["time"].values
            if len(t) >= 2:
                dt = np.diff(t)
                fs_est = 1.0 / np.median(dt[dt>0])
                if np.isfinite(fs_est) and fs_est > 1:
                    self.spin_fs.setValue(float(fs_est))
            self._eeg_df = df
            self.info.emit(f"已加载 EEG CSV：{os.path.basename(path)}，形状 {df.shape}")
            self._draw_all()
        except Exception as e:
            self.info.emit(f"加载失败：{e}")

    def _draw_eeg_and_spec(self):
        ax1, ax2 = self.ax_eeg, self.ax_spec
        ax1.clear(); ax2.clear()
        if self._eeg_df is None or self._eeg_df.empty:
            ax1.text(0.5, 0.5, "未加载 EEG CSV", ha="center", va="center", transform=ax1.transAxes)
            ax2.text(0.5, 0.5, "未加载 EEG CSV", ha="center", va="center", transform=ax2.transAxes)
            return

        df = self._eeg_df.copy()
        time = df["time"].values
        ch_names = [c for c in df.columns if c != "time"]
        if not ch_names:
            ax1.text(0.5, 0.5, "CSV 未找到通道列", ha="center", va="center", transform=ax1.transAxes)
            ax2.text(0.5, 0.5, "CSV 未找到通道列", ha="center", va="center", transform=ax2.transAxes)
            return

        X = df[ch_names].values
        fs = float(self.spin_fs.value())
        # 下采样
        ds = max(1, int(self.spin_down.value()))
        time_ds = time[::ds]
        X_ds = X[::ds, :]

        # 滤波
        fmode = self.cmb_filter.currentText()
        f1 = float(self.spin_f1.value())
        f2 = float(self.spin_f2.value())
        if fmode == "带通(8-30)":
            X_ds = butter_filter(X_ds, fs/ds, 8.0, 30.0, btype='band')
        elif fmode == "低通":
            X_ds = butter_filter(X_ds, fs/ds, None, f2, btype='low')
        elif fmode == "高通":
            X_ds = butter_filter(X_ds, fs/ds, f1, None, btype='high')
        elif fmode == "自定义带通":
            X_ds = butter_filter(X_ds, fs/ds, f1, f2, btype='band')

        # 叠加绘图（通道间加入偏置，避免压在一起）
        offset = 0.0
        for i, name in enumerate(ch_names):
            y = X_ds[:, i]
            ax1.plot(time_ds, y + offset, linewidth=0.8)
            # 根据通道幅值动态偏置
            offset += max(1.0, np.nanmax(np.abs(y)) * 1.2 + 10.0)
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("电位 (μV) [已叠加偏置]")
        ax1.set_title(f"EEG 多通道叠加（共 {len(ch_names)} 通道）")
        ax1.grid(True, alpha=0.2)

        # 功率谱热力（频道×频率）
        # 计算 Welch PSD
        f, pxx = compute_welch_psd(X, fs=fs, nperseg=min(2048, len(X)), noverlap=None)
        if f.size == 0:
            ax2.text(0.5,0.5,"PSD 计算失败", ha="center", va="center", transform=ax2.transAxes)
            return
        # 选择 4~40Hz 范围
        fmin, fmax = 4.0, 40.0
        idx = (f >= fmin) & (f <= fmax)
        f_sel = f[idx]
        pxx_sel = pxx[idx, :]    # shape: [freq, ch]
        # 转 dB
        pxx_db = 10*np.log10(np.maximum(pxx_sel, 1e-12))
        im = ax2.imshow(pxx_db.T, aspect='auto', origin='lower',
                        extent=[f_sel[0], f_sel[-1], 0, len(ch_names)], cmap="viridis")
        ax2.set_yticks(np.arange(len(ch_names)) + 0.5)
        ax2.set_yticklabels(ch_names)
        ax2.set_xlabel("频率 (Hz)"); ax2.set_ylabel("通道")
        ax2.set_title("功率谱热力图（4–40 Hz, dB）")
        self.canvas.fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # ---------- 关闭处理 ----------
    def closeEvent(self, e):
        try:
            self.conn.close()
        except Exception:
            pass
        super().closeEvent(e)
