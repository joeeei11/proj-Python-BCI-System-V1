# -*- coding: utf-8 -*-
# task_module.py
#
# 运动范式（左/右手抓握）—— 回灌统计升级版 + GIF路径可配置（最小改动）
#
# 说明：
# - 四阶段：注视点(+) → 方向提示(←/→) → 运动想象(GIF) → 休息
# - 与主程序/EEG/外设的信号与接口保持不变：
#     info(str), stage(str,int), trial_progress(int,int)
#     notify_trial_result(pred, success, intended_label)
#     notify_device_send(ok, message)
#
# 新增：
# - 左/右手 GIF 路径可通过界面“浏览”选择，并用 QSettings 持久保存
# - 若找不到 GIF，将在“休息”页显示“未找到GIF：绝对路径”以便排查

import os, csv
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QMovie
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QGroupBox, QGridLayout, QStackedLayout,
    QMessageBox, QCheckBox, QFileDialog, QLineEdit
)

# 默认文件名（仍兼容你原有工程的相对路径）
LEFT_GIF_DEFAULT = "assets/left_hand_grasp.gif"
RIGHT_GIF_DEFAULT = "assets/right_hand_grasp.gif"

class StageBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels=[]
        bar = QHBoxLayout(); bar.setSpacing(10); bar.setContentsMargins(0,0,0,0)
        for s in ["注视点","方向提示","运动想象","休息结束"]:
            lab = QLabel(s); lab.setAlignment(Qt.AlignCenter); lab.setObjectName("stageChip")
            self.labels.append(lab); bar.addWidget(lab,1)
        self.setLayout(bar)
    def highlight(self, idx:int):
        for i,lab in enumerate(self.labels):
            lab.setProperty("active", i==idx)
            lab.style().unpolish(lab); lab.style().polish(lab)

class StimulusArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack = QStackedLayout(); self.setLayout(self.stack)
        self.fix = QLabel("+"); self.fix.setAlignment(Qt.AlignCenter); self.fix.setObjectName("fixation")
        self.cue = QLabel("←"); self.cue.setAlignment(Qt.AlignCenter); self.cue.setObjectName("cue")
        self.gif = QLabel(); self.gif.setAlignment(Qt.AlignCenter); self.gif.setObjectName("gif"); self.gif.setScaledContents(True)
        self.movie = QMovie(self)      # 让 QMovie 有持久父对象，避免生命周期问题
        self.gif.setMovie(self.movie)
        self.rest = QLabel("休息"); self.rest.setAlignment(Qt.AlignCenter); self.rest.setObjectName("rest")
        for w in (self.fix,self.cue,self.gif,self.rest): self.stack.addWidget(w)
        self.stack.setCurrentIndex(3)
    def show_fix(self): self.stack.setCurrentIndex(0)
    def show_cue(self, is_left:bool): self.cue.setText("←" if is_left else "→"); self.stack.setCurrentIndex(1)
    def show_gif(self, path:str):
        abspath = os.path.abspath(path or "")
        if (not path) or (not os.path.exists(path)):
            self.rest.setText("未找到GIF：\n"+abspath); self.stack.setCurrentIndex(3); return
        self.movie.stop()
        self.movie.setFileName(path)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.start()
        self.stack.setCurrentIndex(2)
    def show_rest(self, text="休息"):
        if self.movie.state()==QMovie.Running: self.movie.stop()
        self.rest.setText(text); self.stack.setCurrentIndex(3)

class TaskModule(QWidget):
    info = pyqtSignal(str)
    stage = pyqtSignal(str,int)
    trial_progress = pyqtSignal(int,int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("运动任务与范式设计（左/右手抓握）")
        self.resize(900, 640); self.setMinimumSize(820, 580)
        QApplication.instance().setFont(QFont("Microsoft YaHei", 12, QFont.Bold))

        # 配置持久化（用于记住 GIF 路径）
        self.settings = QSettings("NeuroPilot", "TaskModule")
        self.left_gif_path = self.settings.value("left_gif", LEFT_GIF_DEFAULT, type=str)
        self.right_gif_path = self.settings.value("right_gif", RIGHT_GIF_DEFAULT, type=str)

        self.stage_bar = StageBar()

        # 参数设置
        param = QGroupBox("参数设置（任务与时间）"); pg = QGridLayout()
        self.task = QComboBox(); self.task.addItems(["左手抓握","右手抓握"])
        self.fix = QDoubleSpinBox(); self._cfg(self.fix,2.00,0.25,0.25,20.0)
        self.cue = QDoubleSpinBox(); self._cfg(self.cue,1.25,0.25,0.25,10.0)
        self.imag= QDoubleSpinBox(); self._cfg(self.imag,4.00,0.25,1.0,20.0)
        self.rest= QDoubleSpinBox(); self._cfg(self.rest,1.00,0.25,0.5,10.0)
        self.loop = QCheckBox("自动循环")
        self.n_trials = QSpinBox(); self.n_trials.setRange(1,999); self.n_trials.setValue(5)
        self.iti = QDoubleSpinBox(); self._cfg(self.iti,2.00,0.25,0.25,20.0)
        self.n_trials.setEnabled(False); self.iti.setEnabled(False)
        self.loop.toggled.connect(lambda on:(self.n_trials.setEnabled(on), self.iti.setEnabled(on)))

        r=0
        pg.addWidget(QLabel("任务选择"), r,0); pg.addWidget(self.task, r,1); r+=1
        pg.addWidget(QLabel("注视点（秒）"), r,0); pg.addWidget(self.fix, r,1); r+=1
        pg.addWidget(QLabel("方向提示（秒）"), r,0); pg.addWidget(self.cue, r,1); r+=1
        pg.addWidget(QLabel("运动想象（秒）"), r,0); pg.addWidget(self.imag, r,1); r+=1
        pg.addWidget(QLabel("休息（秒）"), r,0); pg.addWidget(self.rest, r,1); r+=1
        pg.addWidget(self.loop, r,0)
        hl=QHBoxLayout(); hl.addWidget(QLabel("试次数")); hl.addWidget(self.n_trials); hl.addSpacing(12); hl.addWidget(QLabel("两次间隔(秒)")); hl.addWidget(self.iti)
        w=QWidget(); w.setLayout(hl); pg.addWidget(w, r,1)
        param.setLayout(pg)

        # —— 新增：刺激素材（GIF）设置（保持最小侵入，不改变原流程）——
        gif_box = QGroupBox("刺激素材（GIF）")
        gg = QGridLayout()
        self.left_edit = QLineEdit(self.left_gif_path)
        self.right_edit = QLineEdit(self.right_gif_path)
        self.btn_browse_left = QPushButton("浏览…")
        self.btn_browse_right = QPushButton("浏览…")
        gg.addWidget(QLabel("左手GIF"), 0, 0); gg.addWidget(self.left_edit, 0, 1); gg.addWidget(self.btn_browse_left, 0, 2)
        gg.addWidget(QLabel("右手GIF"), 1, 0); gg.addWidget(self.right_edit, 1, 1); gg.addWidget(self.btn_browse_right, 1, 2)
        gif_box.setLayout(gg)
        self.btn_browse_left.clicked.connect(lambda: self._pick_gif(side="left"))
        self.btn_browse_right.clicked.connect(lambda: self._pick_gif(side="right"))

        # 刺激与提示
        self.stim = StimulusArea()
        self.tip = QLabel("提示：请选择任务与时间后，点击“开始试次”。"); self.tip.setAlignment(Qt.AlignCenter)

        # 统计
        stat = QGroupBox("统计")
        sh = QHBoxLayout()
        self.t_total = QLabel("总完成：0")
        self.t_succ  = QLabel("预测成功：0")
        self.t_send  = QLabel("发送成功：0")
        self.btn_csv = QPushButton("导出CSV"); self.btn_csv.clicked.connect(self.export_csv)
        for w in (self.t_total,self.t_succ,self.t_send): sh.addWidget(w)
        sh.addStretch(1); sh.addWidget(self.btn_csv); stat.setLayout(sh)

        # 控制按钮
        self.btn_start = QPushButton("开始试次"); self.btn_start.clicked.connect(self.start_trial)
        self.btn_stop  = QPushButton("中止"); self.btn_stop.clicked.connect(self.abort_trial); self.btn_stop.setEnabled(False)

        # —— 布局（左栏：参数+GIF；右栏：刺激显示）——
        left_col = QVBoxLayout()
        left_col.addWidget(param)
        left_col.addWidget(gif_box)
        left_col.addStretch(1)
        left_panel = QWidget(); left_panel.setLayout(left_col)

        root = QVBoxLayout()
        root.addWidget(self.stage_bar); root.addSpacing(8)
        mid = QHBoxLayout(); mid.addStretch(1); mid.addWidget(left_panel); mid.addSpacing(18); mid.addWidget(self.stim,1); mid.addStretch(1)
        root.addLayout(mid,1); root.addWidget(self.tip)
        ctl=QHBoxLayout(); ctl.addStretch(1); ctl.addWidget(self.btn_start); ctl.addWidget(self.btn_stop); ctl.addStretch(1)
        root.addLayout(ctl); root.addWidget(stat); root.addSpacing(6)
        self.setLayout(root)
        self._styles()

        # 状态/统计缓存
        self._running=False; self._timers=[]
        self._loop_left=0; self._iti_ms=0
        self._records=[]   # [时间, 意图, 预测, 是否成功, fix, cue, imag, rest, 发送OK, 设备反馈]
        self._last_idx=None
        self._cnt_total=0; self._cnt_succ=0; self._cnt_send=0

    def _styles(self):
        self.setStyleSheet("""
            QWidget { background:#FFFFFF; color:#323232; font-family:"Microsoft YaHei","微软雅黑",Arial; font-size:14px; }
            QGroupBox { border:1px solid #E6E6E6; border-radius:12px; padding:12px; margin-top:8px; background:#FAFAFA; font-weight:bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding:0 6px; }
            QLabel { font-weight:bold; }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { border:1px solid #D0D0D0; border-radius:8px; padding:6px 10px; background:#F7F7F7; min-width:140px; }
            QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus { border:1px solid #007AFF; background:#FFFFFF; }
            QPushButton { background:#007AFF; color:#FFF; padding:10px 20px; border-radius:10px; font-weight:bold; border:none; min-width:120px; }
            QPushButton:hover { background:#1A84FF; } QPushButton:pressed { background:#0062CC; }
            QPushButton:disabled { background:#E0E0E0; color:#9E9E9E; }
            QLabel#stageChip { background:#F0F0F0; border:1px solid #E6E6E6; border-radius:14px; padding:6px 10px; min-height:28px; }
            QLabel#stageChip[active="true"] { background:#E8F1FF; color:#007AFF; border:1px solid #BFD8FF; }
            QLabel#fixation { font-size:80px; color:#000; }
            QLabel#cue { font-size:80px; color:#000; }
            QLabel#rest { font-size:28px; color:#323232; }
        """)

    def _cfg(self, spin, val, step, lo, hi):
        spin.setDecimals(2); spin.setSingleStep(step); spin.setRange(lo,hi); spin.setValue(val)

    # ===== GIF 路径选择与持久化 =====
    def _pick_gif(self, side: str):
        path, _ = QFileDialog.getOpenFileName(self, "选择 GIF", "", "GIF 文件 (*.gif)")
        if not path: return
        if side == "left":
            self.left_edit.setText(path)
            self.left_gif_path = path
            self.settings.setValue("left_gif", path)
        else:
            self.right_edit.setText(path)
            self.right_gif_path = path
            self.settings.setValue("right_gif", path)

    # ===== 导出 =====
    def export_csv(self):
        if not self._records:
            QMessageBox.information(self,"提示","暂无记录可导出。"); return
        path, _ = QFileDialog.getSaveFileName(self,"导出CSV","MI_trials.csv","CSV Files (*.csv)")
        if not path: return
        with open(path,"w",newline="",encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["时间","意图任务","预测","是否成功","注视点(s)","方向(s)","想象(s)","休息(s)","发送OK","设备反馈"])
            w.writerows(self._records)
        QMessageBox.information(self,"完成",f"已导出：{path}")

    # ===== 回灌入口（主程序调用）=====
    def notify_trial_result(self, pred: str, success: bool, intended_label: str):
        """EEG 模块结束一次试次后，主程序写入：预测/成功"""
        if self._last_idx is None:
            # 理论上 start_trial 时已建记录；若时序抖动，补建一条
            rec = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   intended_label, pred, success,
                   round(self.fix.value(),2), round(self.cue.value(),2),
                   round(self.imag.value(),2), round(self.rest.value(),2),
                   "", ""]
            self._records.append(rec); self._last_idx=len(self._records)-1
        else:
            self._records[self._last_idx][2] = pred
            self._records[self._last_idx][3] = success
        if success:
            self._cnt_succ += 1; self.t_succ.setText(f"预测成功：{self._cnt_succ}")

    def notify_device_send(self, ok: bool, message: str):
        """外设模块一次发送完成/超时后，主程序写入发送结果/设备反馈"""
        if self._last_idx is None: return
        self._records[self._last_idx][8] = "是" if ok else "否"
        self._records[self._last_idx][9] = (message or "").strip()
        if ok:
            self._cnt_send += 1; self.t_send.setText(f"发送成功：{self._cnt_send}")

    # ===== 调度：一次试次 =====
    def start_trial(self):
        if self._running: return
        is_left = (self.task.currentIndex()==0)
        fix_ms=int(self.fix.value()*1000); cue_ms=int(self.cue.value()*1000)
        imag_ms=int(self.imag.value()*1000); rest_ms=int(self.rest.value()*1000)
        if imag_ms < 500:
            QMessageBox.warning(self,"时间过短","运动想象建议不小于 0.5 秒。"); return

        self._loop_left = self.n_trials.value() if self.loop.isChecked() else 1
        self._iti_ms = int(self.iti.value()*1000) if self.loop.isChecked() else 0

        self._running=True; self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._clear_timers()
        self.info.emit("试次开始：注视点阶段"); self.stage.emit("注视点",0); self.stage_bar.highlight(0)
        self.stim.show_fix(); self.tip.setText("提示：注视点（请集中注意力，保持放松）")

        # 先建记录（预测/发送稍后填）
        rec=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "左手抓握" if is_left else "右手抓握", "", "",
             round(self.fix.value(),2), round(self.cue.value(),2),
             round(self.imag.value(),2), round(self.rest.value(),2),
             "", ""]
        self._records.append(rec); self._last_idx=len(self._records)-1

        # 时间轴：t1 cue, t2 imag, t3 rest, t4 end
        t1=fix_ms; t2=t1+cue_ms; t3=t2+imag_ms; t4=t3+rest_ms
        self._add(t1, lambda: self._enter_cue(is_left))
        self._add(t2, self._enter_imag)      # ← 进入想象阶段时再根据当前配置选择 GIF（更稳）
        self._add(t3, self._enter_rest)
        self._add(t4, self._finish_one)

        total = self._loop_left
        self.trial_progress.emit(0,total)

    def _add(self, msec, fn):
        t = QTimer(self); t.setSingleShot(True); t.timeout.connect(fn); t.start(msec); self._timers.append(t)
    def _clear_timers(self):
        for t in self._timers: t.stop(); t.deleteLater()
        self._timers.clear()

    def _enter_cue(self, is_left):
        self.info.emit("方向提示阶段"); self.stage.emit("方向提示",1)
        self.stage_bar.highlight(1); self.stim.show_cue(is_left)
        self.tip.setText("提示：方向提示（← 左 / → 右），请准备开始运动想象")

    def _enter_imag(self):
        self.info.emit("运动想象阶段"); self.stage.emit("运动想象",2); self.stage_bar.highlight(2)
        # 根据当前任务与配置选择 GIF（优先用户选择的路径）
        is_left = (self.task.currentIndex()==0)
        path = (self.left_edit.text().strip() or self.left_gif_path) if is_left else (self.right_edit.text().strip() or self.right_gif_path)
        if not path or not os.path.exists(path):
            # 回退到默认文件名，仍失败则提示绝对路径
            path = LEFT_GIF_DEFAULT if is_left else RIGHT_GIF_DEFAULT
        self.stim.show_gif(path)
        self.tip.setText("提示：运动想象期（请根据GIF想象抓握动作，保持专注）")

    def _enter_rest(self):
        self.info.emit("休息阶段"); self.stage.emit("休息结束",3)
        self.stage_bar.highlight(3); self.stim.show_rest("休息")
        self.tip.setText("提示：休息阶段，请放松")

    def _finish_one(self):
        self._cnt_total += 1; self.t_total.setText(f"总完成：{self._cnt_total}")
        self._loop_left -= 1
        if self._loop_left>0 and self.loop.isChecked():
            self.stim.show_rest(f"休息（下次在 {self.iti.value():.2f}s 后开始）")
            self._add(self._iti_ms, self.start_trial)
        else:
            self._running=False; self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
            self.stim.show_rest("已结束"); self.tip.setText("提示：本次试次已结束。")
            self.info.emit("试次结束")

    def abort_trial(self):
        if not self._running: return
        self._running=False; self._loop_left=0; self._clear_timers()
        self.stim.show_rest("已中止"); self.stage_bar.highlight(3)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.tip.setText("提示：试次已中止。")
