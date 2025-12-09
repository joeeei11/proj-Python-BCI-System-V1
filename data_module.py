# -*- coding: utf-8 -*-
# data_module.py
# 数据存储与分析模块 (Fluent Design Phase 9.5 Visual Polish)
# 职责：数据库交互、EEG 回放分析、训练日志可视化
# 状态：Fixed (Icon Error Resolved)

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QHeaderView, QTableWidgetItem, QSizePolicy
)

# --- 核心：Fluent UI 组件库 ---
from qfluentwidgets import (
    SmoothScrollArea, CardWidget, SimpleCardWidget, ElevatedCardWidget,
    PrimaryPushButton, PushButton, ToolButton,
    ComboBox, DoubleSpinBox, SpinBox, TableWidget,
    TitleLabel, SubtitleLabel, BodyLabel, CaptionLabel, StrongBodyLabel,
    FluentIcon as FIF, IconWidget, InfoBar, InfoBarPosition, theme
)

# --- 绘图依赖 ---
import matplotlib

# 尝试设置字体以适配 Windows UI
matplotlib.rcParams['font.family'] = ['Segoe UI', 'Microsoft YaHei', 'Sans-serif']
matplotlib.rcParams['font.size'] = 9
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- 核心算法库 ---
try:
    from core import dsp
except ImportError:
    # 兜底：运行时应确保 core.dsp 存在
    class dsp:
        @staticmethod
        def butter_filter(d, fs, l, h, order=4): return d

        @staticmethod
        def notch_filter(d, fs, freq=50): return d

        @staticmethod
        def compute_psd(d, fs, nperseg=512, axis=0): return np.array([]), np.array([])


class MplCanvas(FigureCanvas):
    """Matplotlib 画布封装 (适配 Light 主题)"""

    def __init__(self, width=8, height=5, dpi=100):
        # 纯白背景 + 紧凑布局
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.fig.patch.set_facecolor('white')
        super().__init__(self.fig)


class DataAnalyticsPanel(QWidget):
    """
    数据分析面板
    包含：训练日志表格、学习曲线、EEG 波形回放、频谱分析
    """
    info = pyqtSignal(str)

    def __init__(self, db_path="data/neuro_pilot.db", parent=None):
        super().__init__(parent)
        self.setObjectName("DataAnalytics")

        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._check_tables()

        # 内部状态
        self._pending_trial = {}  # 待写入缓存
        self._eeg_df = None  # 当前加载的 EEG 数据
        self._redraw_pending = False

        self._init_ui()

        # 延时加载数据
        QTimer.singleShot(500, self.refresh_table)
        QTimer.singleShot(800, self._draw_all)

    def _check_tables(self):
        """建表 (若不存在)"""
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                session TEXT,
                username TEXT,
                intended_label TEXT,
                predicted TEXT,
                success INTEGER,
                send_ok INTEGER,
                message TEXT,
                fix_s REAL, cue_s REAL, imag_s REAL, rest_s REAL
            )
        """)
        self.conn.commit()

    def _init_ui(self):
        """构建 Fluent UI (Visual Polish)"""
        # 1. 根布局使用 SmoothScrollArea
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = SmoothScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")

        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)

        # 2. 内容布局 (增加呼吸感 Spacing=20, Margins=24)
        self.v_layout = QVBoxLayout(self.content_widget)
        self.v_layout.setContentsMargins(24, 24, 24, 24)
        self.v_layout.setSpacing(20)

        # ==========================================
        # A. 顶部工具栏 (Header Card)
        # ==========================================
        self.header_card = SimpleCardWidget()
        h_layout = QHBoxLayout(self.header_card)
        h_layout.setContentsMargins(24, 16, 24, 16)
        h_layout.setSpacing(16)

        # [FIX] 使用有效的图标
        icon = IconWidget(FIF.MARKET)
        icon.setFixedSize(40, 40)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_lbl = TitleLabel("训练日志与分析", self)
        sub_lbl = CaptionLabel("查看历史训练数据、学习曲线及脑电信号回放", self)
        sub_lbl.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)

        # 按钮组
        self.btn_refresh = PrimaryPushButton(FIF.UPDATE, "刷新", self)
        self.btn_export_csv = PushButton(FIF.SHARE, "导出 CSV", self)
        self.btn_export_json = PushButton(FIF.DOCUMENT, "导出 JSON", self)

        # 统一按钮宽度
        for btn in [self.btn_refresh, self.btn_export_csv, self.btn_export_json]:
            btn.setFixedWidth(120)

        h_layout.addWidget(icon)
        h_layout.addLayout(title_box)
        h_layout.addStretch(1)
        h_layout.addWidget(self.btn_refresh)
        h_layout.addWidget(self.btn_export_csv)
        h_layout.addWidget(self.btn_export_json)

        self.btn_refresh.clicked.connect(self.refresh_table)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)

        self.v_layout.addWidget(self.header_card)

        # ==========================================
        # B. 控制区 (Split Layout)
        # ==========================================
        # 使用 QHBoxLayout 将统计配置和 EEG 配置并排
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(20)

        # --- B1. 统计配置 (左侧) ---
        stat_card = CardWidget(self)
        stat_l = QVBoxLayout(stat_card)
        stat_l.setContentsMargins(20, 20, 20, 20)
        stat_l.setSpacing(12)

        stat_l.addWidget(StrongBodyLabel("📊 统计配置", self))

        row_stat = QHBoxLayout()
        row_stat.setSpacing(12)

        self.cmb_curve = ComboBox(self)
        self.cmb_curve.addItems(["按会话（日）", "按周聚合", "按月聚合"])
        self.cmb_curve.setMinimumWidth(150)

        self.btn_draw = PushButton(FIF.SYNC, "更新图表", self)
        self.btn_draw.clicked.connect(self._draw_all)

        row_stat.addWidget(CaptionLabel("聚合粒度:", self))
        row_stat.addWidget(self.cmb_curve)
        row_stat.addWidget(self.btn_draw)
        stat_l.addLayout(row_stat)

        # T检验标签
        self.lab_ttest = CaptionLabel("T检验 (Welch's): 暂无数据", self)
        self.lab_ttest.setTextColor(QColor("#009FAA"), QColor("#009FAA"))  # Teal color
        stat_l.addWidget(self.lab_ttest)
        stat_l.addStretch(1)  # 顶上去

        ctrl_layout.addWidget(stat_card, 4)  # 权重 4

        # --- B2. EEG 信号设置 (右侧) ---
        eeg_card = CardWidget(self)
        eeg_l = QGridLayout(eeg_card)
        eeg_l.setContentsMargins(20, 20, 20, 20)
        eeg_l.setVerticalSpacing(16)
        eeg_l.setHorizontalSpacing(16)

        eeg_l.addWidget(StrongBodyLabel("🧠 EEG 信号处理", self), 0, 0, 1, 4)

        # Row 1
        self.btn_load_csv = PushButton(FIF.FOLDER, "选择文件", self)
        self.btn_load_csv.clicked.connect(self._load_eeg_csv)
        self.btn_load_csv.setFixedWidth(110)

        self.spin_fs = DoubleSpinBox(self)
        self.spin_fs.setRange(1, 2000)
        self.spin_fs.setValue(250.0)
        self.spin_fs.setMinimumWidth(100)

        self.spin_down = SpinBox(self)
        self.spin_down.setRange(1, 20)
        self.spin_down.setValue(4)
        self.spin_down.setMinimumWidth(80)

        eeg_l.addWidget(self.btn_load_csv, 1, 0)
        eeg_l.addWidget(CaptionLabel("采样率(Hz):", self), 1, 1)
        eeg_l.addWidget(self.spin_fs, 1, 2)
        eeg_l.addWidget(CaptionLabel("下采样:", self), 1, 3)
        eeg_l.addWidget(self.spin_down, 1, 4)

        # Row 2
        self.cmb_filter = ComboBox(self)
        self.cmb_filter.addItems(["不滤波", "带通 (8-30Hz)", "低通 (<30Hz)", "高通 (>8Hz)", "自定义"])
        self.cmb_filter.setMinimumWidth(110)

        self.spin_f1 = DoubleSpinBox(self)
        self.spin_f1.setValue(8.0)
        self.spin_f1.setMinimumWidth(80)

        self.spin_f2 = DoubleSpinBox(self)
        self.spin_f2.setValue(30.0)
        self.spin_f2.setMinimumWidth(80)

        eeg_l.addWidget(self.cmb_filter, 2, 0)
        eeg_l.addWidget(CaptionLabel("低频截止:", self), 2, 1)
        eeg_l.addWidget(self.spin_f1, 2, 2)
        eeg_l.addWidget(CaptionLabel("高频截止:", self), 2, 3)
        eeg_l.addWidget(self.spin_f2, 2, 4)

        ctrl_layout.addWidget(eeg_card, 6)  # 权重 6
        self.v_layout.addLayout(ctrl_layout)

        # ==========================================
        # C. 可视化画布 (Elevated Card - 视觉重心)
        # ==========================================
        self.chart_card = ElevatedCardWidget(self)
        self.chart_card.setMinimumHeight(600)
        chart_l = QVBoxLayout(self.chart_card)
        chart_l.setContentsMargins(0, 0, 0, 0)

        self.canvas = MplCanvas(width=10, height=10, dpi=100)
        self.ax_curve = self.canvas.fig.add_subplot(2, 2, 1)
        self.ax_cm = self.canvas.fig.add_subplot(2, 2, 2)
        self.ax_eeg = self.canvas.fig.add_subplot(2, 2, 3)
        self.ax_spec = self.canvas.fig.add_subplot(2, 2, 4)

        chart_l.addWidget(self.canvas)
        self.v_layout.addWidget(self.chart_card)

        # ==========================================
        # D. 数据表格 (CardWidget)
        # ==========================================
        self.table_card = CardWidget(self)
        table_l = QVBoxLayout(self.table_card)
        table_l.setContentsMargins(20, 20, 20, 20)
        table_l.setSpacing(12)

        table_header = QHBoxLayout()
        table_title = StrongBodyLabel("📋 最近试次记录", self)
        table_header.addWidget(table_title)
        table_header.addStretch(1)
        table_l.addLayout(table_header)

        self.table = TableWidget(self)
        self.table.setBorderVisible(True)
        self.table.setBorderRadius(8)
        self.table.setWordWrap(False)
        self.table.setColumnCount(12)
        headers = [
            "时间", "会话", "用户", "意图", "预测", "成功", "发送", "信息",
            "注视(s)", "提示(s)", "想象(s)", "休息(s)"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(280)

        table_l.addWidget(self.table)
        self.v_layout.addWidget(self.table_card)

    # ======================================================
    # 辅助功能：Matplotlib 风格化
    # ======================================================
    def _style_axis(self, ax, title=""):
        """统一设置图表风格：去边框、柔和网格、深灰字体"""
        ax.set_title(title, fontsize=10, fontweight='bold', color='#333333', pad=10)

        # 去除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 柔和的坐标轴颜色
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['bottom'].set_color('#E0E0E0')

        # 字体颜色
        ax.tick_params(axis='x', colors='#606060', labelsize=8)
        ax.tick_params(axis='y', colors='#606060', labelsize=8)
        ax.yaxis.label.set_color('#606060')
        ax.xaxis.label.set_color('#606060')

        # 虚线网格
        ax.grid(True, linestyle='--', alpha=0.3, color='#C0C0C0')

    # ======================================================
    # 业务逻辑
    # ======================================================

    def refresh_table(self):
        df = self._read_df()
        self._fill_table(df)
        self.info.emit(f"数据已刷新: 共 {len(df)} 条")
        self._debounced_draw()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "trials.csv", "CSV Files (*.csv)")
        if not path: return
        df = self._read_df()
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self._show_msg("导出成功", f"已保存至 {path}", success=True)
        except Exception as e:
            self._show_msg("导出失败", str(e), success=False)

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", "trials.json", "JSON Files (*.json)")
        if not path: return
        df = self._read_df()
        try:
            df.to_json(path, orient="records", force_ascii=False, indent=2)
            self._show_msg("导出成功", f"已保存至 {path}", success=True)
        except Exception as e:
            self._show_msg("导出失败", str(e), success=False)

    def _show_msg(self, title, content, success=True):
        self.info.emit(f"{title}: {content}")
        if success:
            InfoBar.success(
                title=title, content=content,
                orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT,
                duration=2000, parent=self
            )
        else:
            InfoBar.error(
                title=title, content=content,
                orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP_RIGHT,
                duration=2000, parent=self
            )

    def _read_df(self):
        try:
            df = pd.read_sql_query("SELECT * FROM trials ORDER BY id DESC", self.conn)
        except Exception:
            cols = ["ts", "session", "username", "intended_label", "predicted", "success",
                    "send_ok", "message", "fix_s", "cue_s", "imag_s", "rest_s"]
            df = pd.DataFrame(columns=cols)
        return df

    def _fill_table(self, df: pd.DataFrame):
        self.table.setRowCount(0)
        if df is None or df.empty: return

        # 字段兼容
        cols_map = {
            "时间": "ts" if "ts" in df.columns else "timestamp",
            "会话": "session" if "session" in df.columns else "session_id",
            "用户": "username" if "username" in df.columns else "subject_name",
            "意图": "intended_label",
            "预测": "predicted" if "predicted" in df.columns else "predicted_label",
            "成功": "success" if "success" in df.columns else "is_success",
            "发送": "send_ok" if "send_ok" in df.columns else "send_status",
            "信息": "message" if "message" in df.columns else "device_msg",
            "注视": "fix_s" if "fix_s" in df.columns else "fix_duration",
            "提示": "cue_s" if "cue_s" in df.columns else "cue_duration",
            "想象": "imag_s" if "imag_s" in df.columns else "imag_duration",
            "休息": "rest_s" if "rest_s" in df.columns else "rest_duration"
        }

        headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]

        self.table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, key in enumerate(headers):
                col_name = cols_map.get(key, "")
                val = row.get(col_name, "")
                # 格式化
                if key in ["成功", "发送"]:
                    try:
                        v_int = int(val)
                        val = "是" if v_int == 1 else "否"
                    except:
                        pass
                elif pd.isna(val):
                    val = ""
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)

    # --- 数据回灌 ---
    def notify_trial_started(self, username, intended, fix, cue, imag, rest):
        self._pending_trial = dict(
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session=datetime.now().strftime("%Y-%m-%d"),
            username=username,
            intended_label=intended,
            predicted=None, success=None, send_ok=None, message=None,
            fix_s=float(fix), cue_s=float(cue), imag_s=float(imag), rest_s=float(rest)
        )

    def notify_trial_result(self, predicted, success):
        if not self._pending_trial: return
        self._pending_trial["predicted"] = predicted
        self._pending_trial["success"] = 1 if success else 0

    def notify_device_send(self, send_ok, message):
        if not self._pending_trial: return
        self._pending_trial["send_ok"] = 1 if send_ok else 0
        self._pending_trial["message"] = str(message)

        # 写入 DB
        try:
            keys = list(self._pending_trial.keys())
            vals = list(self._pending_trial.values())
            placeholders = ",".join(["?"] * len(keys))
            columns = ",".join(keys)

            sql = f"INSERT INTO trials ({columns}) VALUES ({placeholders})"
            c = self.conn.cursor()
            c.execute(sql, vals)
            self.conn.commit()
            self.info.emit("记录已保存")
        except Exception as e:
            self.info.emit(f"保存失败: {e}")
        finally:
            self._pending_trial = {}
            self.refresh_table()

    # --- 绘图 (Visual Polish) ---
    def _debounced_draw(self):
        if self._redraw_pending: return
        self._redraw_pending = True
        QTimer.singleShot(200, self._draw_all)

    def _draw_all(self):
        self._redraw_pending = False
        if not self.isVisible(): return

        for ax in [self.ax_curve, self.ax_cm, self.ax_eeg, self.ax_spec]:
            ax.clear()

        df = self._read_df()

        # 1. 学习曲线
        self._plot_learning_curve(df)
        # 2. 统计
        self._plot_stats(df)
        # 3. EEG
        self._draw_eeg_visuals()

        self.canvas.draw_idle()

    def _plot_learning_curve(self, df):
        self._style_axis(self.ax_curve, "Learning Curve (Accuracy)")

        if df is None or df.empty or "success" not in df.columns:
            self.ax_curve.text(0.5, 0.5, "No Data", ha='center', color='#999999')
            return

        try:
            df_ok = df.copy()
            ts_col = "ts" if "ts" in df_ok.columns else "timestamp"
            df_ok[ts_col] = pd.to_datetime(df_ok[ts_col], errors='coerce')
            df_ok["success"] = pd.to_numeric(df_ok["success"], errors='coerce').fillna(0)

            idx = self.cmb_curve.currentIndex()
            if idx == 0:
                grp = df_ok.groupby("session" if "session" in df_ok.columns else "session_id")
            elif idx == 1:
                grp = df_ok.groupby(df_ok[ts_col].dt.to_period("W").dt.start_time)
            else:
                grp = df_ok.groupby(df_ok[ts_col].dt.to_period("M").dt.start_time)

            x_vals, y_vals = [], []
            for k, g in grp:
                if len(g) > 0:
                    x_vals.append(str(k)[:10])
                    y_vals.append(g["success"].mean())

            if x_vals:
                # 使用现代的蓝色 (#009FAA)
                self.ax_curve.plot(x_vals, y_vals, marker="o", color="#009FAA", linewidth=2.5, markersize=6)
                self.ax_curve.fill_between(x_vals, y_vals, alpha=0.1, color="#009FAA")
                self.ax_curve.set_ylim(0, 1.1)
                self.ax_curve.tick_params(axis='x', rotation=30)
        except Exception as e:
            self.ax_curve.text(0.5, 0.5, f"Err: {e}", ha='center', fontsize=8)

    def _plot_stats(self, df):
        self._style_axis(self.ax_cm, "Confusion Matrix")

        if df is None or df.empty: return

        # T-Test
        try:
            col_intent = "intended_label"
            col_succ = "success"
            lefts = df[df[col_intent].astype(str).str.contains("左|Left", na=False)][col_succ]
            rights = df[df[col_intent].astype(str).str.contains("右|Right", na=False)][col_succ]
            lefts = pd.to_numeric(lefts, errors='coerce')
            rights = pd.to_numeric(rights, errors='coerce')

            if len(lefts) > 1 and len(rights) > 1:
                res = ttest_ind(lefts, rights, equal_var=False)
                self.lab_ttest.setText(f"Welch t-test: p={res.pvalue:.4f}")
            else:
                self.lab_ttest.setText("T检验: 样本不足")
        except:
            pass

        # Confusion Matrix
        try:
            y_true = df["intended_label"].map(lambda x: 0 if "左" in str(x) or "Left" in str(x) else 1)
            col_pred = "predicted" if "predicted" in df.columns else "predicted_label"
            y_pred = df[col_pred].map(
                lambda x: 0 if "left" in str(x).lower() else (1 if "right" in str(x).lower() else -1))
            mask = (y_pred != -1)
            if mask.any():
                cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
                # 使用 GnBu 渐变色
                self.ax_cm.imshow(cm, cmap="GnBu")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        self.ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color='#333333',
                                        fontweight='bold')
                self.ax_cm.set_xticks([0, 1]);
                self.ax_cm.set_xticklabels(["L", "R"])
                self.ax_cm.set_yticks([0, 1]);
                self.ax_cm.set_yticklabels(["L", "R"])
        except:
            pass

    def _load_eeg_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV", "", "CSV (*.csv)")
        if path:
            try:
                self._eeg_df = pd.read_csv(path)
                self.info.emit(f"已加载: {os.path.basename(path)}")
                self._draw_all()
            except Exception as e:
                self.info.emit(f"加载失败: {e}")

    def _draw_eeg_visuals(self):
        self._style_axis(self.ax_eeg, "EEG Waveforms")
        self._style_axis(self.ax_spec, "PSD (dB)")

        if self._eeg_df is None:
            self.ax_eeg.text(0.5, 0.5, "No EEG Loaded", ha='center', color='#999999')
            self.ax_spec.text(0.5, 0.5, "No Data", ha='center', color='#999999')
            return

        cols = [c for c in self._eeg_df.columns if 'time' not in c.lower()]
        if not cols: return

        data = self._eeg_df[cols].values
        fs = self.spin_fs.value()

        # Filter
        f_mode = self.cmb_filter.currentText()
        if "带通" in f_mode or "自定义" in f_mode:
            low = 8.0 if "自定义" not in f_mode else self.spin_f1.value()
            high = 30.0 if "自定义" not in f_mode else self.spin_f2.value()
            data = dsp.butter_filter(data, fs, low, high, order=4)
        elif "低通" in f_mode:
            data = dsp.butter_filter(data, fs, None, self.spin_f2.value())
        elif "高通" in f_mode:
            data = dsp.butter_filter(data, fs, self.spin_f1.value(), None)

        ds = int(self.spin_down.value())
        data_ds = data[::ds]
        t = np.arange(len(data_ds)) / (fs / ds)

        offset = 0
        # 现代配色盘
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6', '#3498DB']

        for i, col in enumerate(cols):
            y = data_ds[:, i]
            amp = np.percentile(np.abs(y), 95) * 2.0
            if amp < 1: amp = 10

            c = colors[i % len(colors)]
            self.ax_eeg.plot(t, y + offset, lw=0.8, color=c, alpha=0.9)
            self.ax_eeg.text(t[0], offset, col, fontsize=8, ha='right', va='center', color='#606060')
            offset += amp

        # PSD
        f, pxx = dsp.compute_psd(data, fs=fs, nperseg=512, axis=0)
        if len(f) > 0:
            idx = (f >= 4) & (f <= 40)
            f_sel = f[idx]
            pxx_sel = pxx[idx, :].T
            im = self.ax_spec.imshow(10 * np.log10(pxx_sel + 1e-9), aspect='auto',
                                     extent=[f_sel[0], f_sel[-1], 0, len(cols)],
                                     origin='lower', cmap='viridis')  # Viridis is good
            self.ax_spec.set_yticks(np.arange(len(cols)) + 0.5)
            self.ax_spec.set_yticklabels(cols)
            self.canvas.fig.colorbar(im, ax=self.ax_spec, fraction=0.046, pad=0.04)

    def closeEvent(self, e):
        if self.conn:
            self.conn.close()
        super().closeEvent(e)