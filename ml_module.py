# -*- coding: utf-8 -*-
# ml_module.py
#
# 模型训练与优化（最小侵入版）—— 支持：
# 1) 离线训练 + 网格搜索 + 交叉验证 + 可视化（混淆矩阵 / ROC）
# 2) 可选：学习曲线（Learning Curve）
# 3) 可选：特征选择（SelectKBest / PCA）
# 4) 可选：批量对比实验（SVM-RBF / SVM-Linear / KNN / LR / RF）
#
# - 不修改 EEG/范式/外设 既有逻辑，仅作为新增标签页
# - 数据来源：CSV（特征表，含 label 列）或 生成演示数据
# - 算法：全部 scikit-learn；不引入 TensorFlow
# - 与主程序联动：在 main.py 中实例化 MLTrainerPanel() 加为一个 tab，将 info 信号接入状态栏与日志

import os
import io
import pickle
import logging
import numpy as np

# 允许没有 pandas 时给出更明确提示（多数环境已装）
try:
    import pandas as pd
except Exception as _e:
    pd = None

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QCheckBox
)

# Matplotlib 嵌入
import matplotlib
# 尽量用微软雅黑，减少中文缺字警告
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

APPLE_BLUE = "#007AFF"
DARK_TEXT = "#323232"


def _parse_param_grid(grid_text: str):
    """
    解析简易网格字符串 -> dict
    输入示例： "C=0.1,1,10; gamma=scale,auto"
    返回：{"C":[0.1,1,10], "gamma":["scale","auto"]}
    """
    grid = {}
    if not grid_text or not grid_text.strip():
        return grid
    parts = grid_text.split(";")
    for p in parts:
        p = p.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        vals = []
        for token in v.split(","):
            token = token.strip()
            if token == "":
                continue
            # 数字自动转型
            try:
                if "." in token:
                    vals.append(float(token))
                else:
                    vals.append(int(token))
            except Exception:
                vals.append(token)
        if vals:
            grid[k] = vals
    return grid


class MplCanvas(FigureCanvas):
    """
    2x2 子图：
    [0,0] 混淆矩阵    [0,1] ROC
    [1,0] 学习曲线    [1,1] 批量对比
    """
    def __init__(self, width=9.5, height=6.8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax_cm = fig.add_subplot(2, 2, 1)
        self.ax_roc = fig.add_subplot(2, 2, 2)
        self.ax_lc = fig.add_subplot(2, 2, 3)
        self.ax_cmp = fig.add_subplot(2, 2, 4)
        super().__init__(fig)


class MLTrainerPanel(QWidget):
    """
    模型训练与优化面板
    - 对外信号：info(str)
    """
    info = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self._log = logging.getLogger("NeuroPilot.ML")

        self.df = None   # pandas DataFrame
        self.X = None    # np.ndarray (n_samples, n_features)
        self.y = None    # np.ndarray (n_samples,)
        self.classes_ = None
        self.model = None  # 训练好的 sklearn Pipeline
        self._last_split = None  # (Xtr, Xte, ytr, yte)

        self._build_ui()
        self._apply_styles()

    # ---------------- UI 搭建 ----------------
    def _build_ui(self):
        # —— 数据源区 ——
        box_data = QGroupBox("数据源")
        g = QGridLayout()
        self.btn_load_csv = QPushButton("导入 CSV")
        self.btn_demo = QPushButton("生成演示数据")
        self.ed_target = QLineEdit("label")
        self.ed_features = QLineEdit("")  # 留空=自动：除目标列外全部为特征
        self.split_spin = QDoubleSpinBox(); self.split_spin.setRange(0.1, 0.9); self.split_spin.setSingleStep(0.05); self.split_spin.setValue(0.2)
        self.btn_preview = QPushButton("预览/检查")
        r = 0
        g.addWidget(QLabel("目标列"), r, 0); g.addWidget(self.ed_target, r, 1); r += 1
        g.addWidget(QLabel("特征列（逗号分隔，空=自动）"), r, 0); g.addWidget(self.ed_features, r, 1, 1, 3); r += 1
        g.addWidget(QLabel("测试集比例"), r, 0); g.addWidget(self.split_spin, r, 1)
        g.addWidget(self.btn_load_csv, r, 2); g.addWidget(self.btn_demo, r, 3); r += 1
        g.addWidget(self.btn_preview, r, 0, 1, 4)
        box_data.setLayout(g)

        # —— 算法与参数区 ——
        box_algo = QGroupBox("算法与参数")
        ag = QGridLayout()
        self.cmb_algo = QComboBox()
        self.cmb_algo.addItems(["SVM (RBF)", "SVM (Linear)", "KNN", "LogisticRegression", "RandomForest", "深度学习（占位，禁用）"])
        self.cmb_algo.setCurrentIndex(0)
        self.cmb_algo.currentIndexChanged.connect(self._on_algo_changed)

        self.ed_grid = QLineEdit("C=0.1,1,10; gamma=scale,auto")  # 默认适配 SVM(RBF)
        self.cv_spin = QSpinBox(); self.cv_spin.setRange(2, 20); self.cv_spin.setValue(5)
        self.cmb_score = QComboBox(); self.cmb_score.addItems(["accuracy", "f1_macro", "roc_auc_ovr"])
        self.chk_standardize = QCheckBox("标准化(Std)"); self.chk_standardize.setChecked(True)

        r = 0
        ag.addWidget(QLabel("算法"), r, 0); ag.addWidget(self.cmb_algo, r, 1, 1, 3); r += 1
        ag.addWidget(QLabel("参数网格"), r, 0); ag.addWidget(self.ed_grid, r, 1, 1, 3); r += 1
        ag.addWidget(QLabel("交叉验证折数"), r, 0); ag.addWidget(self.cv_spin, r, 1)
        ag.addWidget(QLabel("评分指标"), r, 2); ag.addWidget(self.cmb_score, r, 3); r += 1
        ag.addWidget(self.chk_standardize, r, 0, 1, 4); r += 1
        box_algo.setLayout(ag)

        # —— 特征选择（可选） ——
        box_feat = QGroupBox("特征选择（可选）")
        fg = QGridLayout()
        self.chk_kbest = QCheckBox("SelectKBest")
        self.cmb_kbest_score = QComboBox(); self.cmb_kbest_score.addItems(["f_classif", "mutual_info"])
        self.spin_k = QSpinBox(); self.spin_k.setRange(1, 4096); self.spin_k.setValue(20)

        self.chk_pca = QCheckBox("PCA")
        self.spin_pca = QSpinBox(); self.spin_pca.setRange(1, 4096); self.spin_pca.setValue(10)

        r = 0
        fg.addWidget(self.chk_kbest, r, 0); fg.addWidget(QLabel("打分函数"), r, 1); fg.addWidget(self.cmb_kbest_score, r, 2); fg.addWidget(QLabel("k"), r, 3); fg.addWidget(self.spin_k, r, 4); r += 1
        fg.addWidget(self.chk_pca, r, 0); fg.addWidget(QLabel("n_components"), r, 1); fg.addWidget(self.spin_pca, r, 2)
        box_feat.setLayout(fg)

        # —— 训练与评估 ——
        box_train = QGroupBox("训练与评估")
        tg = QGridLayout()
        self.btn_train = QPushButton("开始训练")
        self.btn_save = QPushButton("保存模型")
        self.btn_load = QPushButton("加载模型并评估")
        self.btn_lc = QPushButton("绘制学习曲线")          # 新增
        self.txt_report = QTextEdit(); self.txt_report.setReadOnly(True)
        self.canvas = MplCanvas()
        tg.addWidget(self.btn_train, 0, 0)
        tg.addWidget(self.btn_save, 0, 1)
        tg.addWidget(self.btn_load, 0, 2)
        tg.addWidget(self.btn_lc,   0, 3)
        tg.addWidget(self.txt_report, 1, 0, 1, 4)
        tg.addWidget(self.canvas, 2, 0, 1, 4)
        box_train.setLayout(tg)

        # —— 批量对比实验（可选） ——
        box_cmp = QGroupBox("批量对比实验（可选）")
        cg = QGridLayout()
        self.chk_cmp_svm_rbf = QCheckBox("SVM-RBF"); self.chk_cmp_svm_rbf.setChecked(True)
        self.chk_cmp_svm_lin = QCheckBox("SVM-Linear")
        self.chk_cmp_knn = QCheckBox("KNN")
        self.chk_cmp_lr = QCheckBox("LogisticRegression")
        self.chk_cmp_rf = QCheckBox("RandomForest")
        self.cmb_cmp_metric = QComboBox(); self.cmb_cmp_metric.addItems(["accuracy", "f1_macro"])
        self.btn_cmp = QPushButton("运行对比")
        row = 0
        cg.addWidget(QLabel("对比算法"), row, 0)
        cg.addWidget(self.chk_cmp_svm_rbf, row, 1)
        cg.addWidget(self.chk_cmp_svm_lin, row, 2)
        cg.addWidget(self.chk_cmp_knn, row, 3)
        cg.addWidget(self.chk_cmp_lr, row, 4)
        cg.addWidget(self.chk_cmp_rf, row, 5)
        row += 1
        cg.addWidget(QLabel("评分指标"), row, 0); cg.addWidget(self.cmb_cmp_metric, row, 1)
        cg.addWidget(self.btn_cmp, row, 5)
        box_cmp.setLayout(cg)

        # —— 总布局 ——
        root = QVBoxLayout()
        root.addWidget(box_data)
        root.addWidget(box_algo)
        root.addWidget(box_feat)
        root.addWidget(box_train)
        root.addWidget(box_cmp)
        self.setLayout(root)

        # —— 事件 ——
        self.btn_load_csv.clicked.connect(self._load_csv)
        self.btn_demo.clicked.connect(self._gen_demo)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_train.clicked.connect(self._train)
        self.btn_save.clicked.connect(self._save_model)
        self.btn_load.clicked.connect(self._load_model)
        self.btn_lc.clicked.connect(self._draw_learning_curve)
        self.btn_cmp.clicked.connect(self._run_comparison)

        os.makedirs("data/models", exist_ok=True)

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
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                padding: 6px 10px;
                background: #F7F7F7;
                min-width: 140px;
            }}
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
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
            QTextEdit {{
                border: 1px solid #E6E6E6;
                border-radius: 8px;
                padding: 8px;
                background: #FFFFFF;
            }}
        """)

    # ---------------- 数据处理 ----------------
    def _load_csv(self):
        if pd is None:
            QMessageBox.critical(self, "缺少依赖", "当前环境未安装 pandas，请先安装：pip install pandas")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择特征CSV", "data", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            self.info.emit(f"CSV已载入：{os.path.basename(path)}，形状={self.df.shape}")
            self._extract_Xy()
        except Exception as e:
            QMessageBox.critical(self, "载入失败", str(e))
            self.info.emit(f"载入失败：{e}")

    def _gen_demo(self):
        """生成演示数据：两类可分样本，模拟CSP后特征空间"""
        n = 300
        rng = np.random.RandomState(0)
        mu0 = np.zeros(12)
        mu1 = np.r_[np.ones(6)*0.8, np.ones(6)*-0.8]
        cov = 0.3*np.eye(12)
        X0 = rng.multivariate_normal(mu0, cov, size=n//2)
        X1 = rng.multivariate_normal(mu1, cov, size=n//2)
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.array([0]*(n//2) + [1]*(n//2))
        if pd is None:
            # 无 pandas：直接缓存为 X/y
            self.df = None
            self.X, self.y = X, y
            self.classes_ = np.array([0, 1])
            self.info.emit("已生成演示数据（无pandas模式）：300×12 + label")
            return
        cols = [f"f{i+1}" for i in range(X.shape[1])]
        self.df = pd.DataFrame(X, columns=cols)
        self.df["label"] = y
        self.info.emit("已生成演示数据：300×12 + label")
        self._extract_Xy()

    def _preview(self):
        if self.df is None:
            QMessageBox.information(self, "提示", "请先导入CSV或生成演示数据。")
            return
        buf = io.StringIO()
        self.df.head(10).to_string(buf, index=False)
        QMessageBox.information(self, "数据预览（前10行）", buf.getvalue())

    def _extract_Xy(self):
        if self.df is None:
            return
        target = self.ed_target.text().strip() or "label"
        if target not in self.df.columns:
            raise ValueError(f"目标列 '{target}' 不存在。")
        feats_txt = self.ed_features.text().strip()
        if feats_txt:
            feats = [c.strip() for c in feats_txt.split(",") if c.strip()]
        else:
            feats = [c for c in self.df.columns if c != target]
        if not feats:
            raise ValueError("未找到任何特征列。")
        X = self.df[feats].values.astype(np.float32)
        y_raw = self.df[target].values
        classes, y_enc = np.unique(y_raw, return_inverse=True)
        self.X, self.y = X, y_enc
        self.classes_ = classes
        self.info.emit(f"特征/标签已抽取：X={X.shape}, y={y_enc.shape}，类别={list(classes)}")

    # ---------------- Pipeline 构建 ----------------
    def _build_feature_steps(self):
        """根据UI构建特征选择/降维步骤"""
        steps = []
        # SelectKBest
        if self.chk_kbest.isChecked():
            k = int(self.spin_k.value())
            score_name = self.cmb_kbest_score.currentText()
            score_func = f_classif if score_name == "f_classif" else mutual_info_classif
            steps.append(("select", SelectKBest(score_func=score_func, k=k)))
        # 标准化
        if self.chk_standardize.isChecked():
            steps.append(("scaler", StandardScaler()))
        # PCA
        if self.chk_pca.isChecked():
            n_comp = int(self.spin_pca.value())
            steps.append(("pca", PCA(n_components=n_comp, random_state=0)))
        return steps

    def _on_algo_changed(self, idx: int):
        name = self.cmb_algo.currentText()
        if name == "SVM (RBF)":
            self.ed_grid.setText("C=0.1,1,10; gamma=scale,auto")
        elif name == "SVM (Linear)":
            self.ed_grid.setText("C=0.1,1,10")
        elif name == "KNN":
            self.ed_grid.setText("n_neighbors=3,5,7,9; weights=uniform,distance")
        elif name == "LogisticRegression":
            self.ed_grid.setText("C=0.1,1,10; penalty=l2; solver=lbfgs")
        elif name == "RandomForest":
            self.ed_grid.setText("n_estimators=100,200; max_depth=3,5,7")
        else:
            self.ed_grid.setText("（深度学习占位，禁用）")

    def _build_estimator_and_grid(self):
        """根据算法选择，构造 Pipeline(est) 与 参数网格"""
        name = self.cmb_algo.currentText()
        grid_user = _parse_param_grid(self.ed_grid.text())
        feat_steps = self._build_feature_steps()

        if name == "SVM (RBF)":
            est = Pipeline(feat_steps + [("clf", SVC(kernel="rbf", probability=True, random_state=0))])
            grid = {f"clf__{k}": v for k, v in grid_user.items()} if grid_user else {"clf__C":[1.0], "clf__gamma":["scale"]}

        elif name == "SVM (Linear)":
            est = Pipeline(feat_steps + [("clf", SVC(kernel="linear", probability=True, random_state=0))])
            grid = {f"clf__{k}": v for k, v in grid_user.items()} if grid_user else {"clf__C":[1.0]}

        elif name == "KNN":
            est = Pipeline(feat_steps + [("clf", KNeighborsClassifier())])
            grid = {f"clf__{k}": v for k, v in grid_user.items()} if grid_user else {"clf__n_neighbors":[5]}

        elif name == "LogisticRegression":
            est = Pipeline(feat_steps + [("clf", LogisticRegression(max_iter=300, random_state=0))])
            grid = {f"clf__{k}": v for k, v in grid_user.items()} if grid_user else {"clf__C":[1.0], "clf__solver":["lbfgs"]}

        elif name == "RandomForest":
            est = Pipeline(feat_steps + [("clf", RandomForestClassifier(random_state=0))])
            grid = {f"clf__{k}": v for k, v in grid_user.items()} if grid_user else {"clf__n_estimators":[200], "clf__max_depth":[5]}

        else:
            raise RuntimeError("深度学习（占位）当前禁用。")
        return est, grid

    # ---------------- 训练/评估/可视化 ----------------
    def _clear_axes(self):
        self.canvas.ax_cm.clear()
        self.canvas.ax_roc.clear()
        self.canvas.ax_lc.clear()
        # 对比图不在每次训练时清空，保留最近一次对比结果
        self.canvas.draw()

    def _plot_confusion_matrix(self, cm, labels):
        ax = self.canvas.ax_cm
        ax.clear()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("混淆矩阵")
        ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        # 标注格子数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        self.canvas.draw()

    def _plot_roc(self, Xte, yte, model=None):
        ax = self.canvas.ax_roc
        ax.clear()
        mdl = model if model is not None else self.model
        if mdl is None:
            ax.text(0.5, 0.5, "无模型", ha="center", va="center")
            self.canvas.draw(); return

        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(Xte)
            if proba.ndim == 2 and proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(yte, proba[:, 1])
                ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
            else:
                # 多分类：One-vs-rest
                for k in range(proba.shape[1]):
                    fpr, tpr, _ = roc_curve((yte == k).astype(int), proba[:, k])
                    ax.plot(fpr, tpr, label=f"class {k} AUC={auc(fpr, tpr):.3f}")
        else:
            ax.text(0.5, 0.5, "当前算法不支持 ROC（缺少 predict_proba）", ha="center", va="center")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC 曲线")
        ax.legend(loc="lower right", fontsize=8)
        self.canvas.draw()

    def _train(self):
        try:
            if self.X is None or self.y is None:
                QMessageBox.information(self, "提示", "请先导入CSV或生成演示数据。")
                return

            test_size = float(self.split_spin.value())
            Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y, random_state=0)
            self._last_split = (Xtr, Xte, ytr, yte)

            est, grid = self._build_estimator_and_grid()
            cv = int(self.cv_spin.value())
            scoring = self.cmb_score.currentText()

            self.info.emit(f"开始训练：算法={self.cmb_algo.currentText()}，CV={cv}，评分={scoring}")
            self.txt_report.clear()
            self._clear_axes()

            gs = GridSearchCV(estimator=est, param_grid=grid, scoring=scoring,
                              cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0),
                              n_jobs=-1)
            gs.fit(Xtr, ytr)

            self.model = gs.best_estimator_
            best_params = gs.best_params_
            best_score = gs.best_score_

            # 测试集评估
            ypred = self.model.predict(Xte)
            report = classification_report(yte, ypred, target_names=[str(c) for c in self.classes_])
            cm = confusion_matrix(yte, ypred)

            out = []
            out.append(f"最优参数：{best_params}")
            out.append(f"CV 最优分数（{scoring}）：{best_score:.4f}")
            out.append("—— 测试集分类报告 ——")
            out.append(report)
            self.txt_report.setPlainText("\n".join(out))

            # 绘图：混淆矩阵 + ROC
            self._plot_confusion_matrix(cm, [str(c) for c in self.classes_])
            self._plot_roc(Xte, yte)

            self.info.emit("训练完成。已在右侧展示混淆矩阵与ROC。")

        except Exception as e:
            QMessageBox.critical(self, "训练失败", str(e))
            self.info.emit(f"训练失败：{e}")

    def _draw_learning_curve(self):
        """使用当前算法与特征步骤绘制学习曲线（可选功能按钮）"""
        try:
            if self.X is None or self.y is None:
                QMessageBox.information(self, "提示", "请先导入CSV或生成演示数据。")
                return

            est, _ = self._build_estimator_and_grid()
            cv = int(self.cv_spin.value())
            scoring = self.cmb_score.currentText()
            # 训练样本比例 5 个点（10%->100%）
            train_sizes = np.linspace(0.1, 1.0, 5)

            self.info.emit("开始绘制学习曲线...")
            self.canvas.ax_lc.clear()
            train_sizes_abs, train_scores, valid_scores = learning_curve(
                est, self.X, self.y,
                train_sizes=train_sizes,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0),
                scoring=scoring,
                n_jobs=-1,
                shuffle=True,
                random_state=0 if "random_state" in est.get_params() else None
            )
            tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
            va_mean, va_std = valid_scores.mean(axis=1), valid_scores.std(axis=1)

            ax = self.canvas.ax_lc
            ax.plot(train_sizes_abs, tr_mean, marker="o", label="训练分数")
            ax.fill_between(train_sizes_abs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)
            ax.plot(train_sizes_abs, va_mean, marker="s", label="验证分数")
            ax.fill_between(train_sizes_abs, va_mean - va_std, va_mean + va_std, alpha=0.2)
            ax.set_xlabel("训练样本数")
            ax.set_ylabel(scoring)
            ax.set_title("学习曲线")
            ax.legend()
            self.canvas.draw()
            self.info.emit("学习曲线绘制完成。")

        except Exception as e:
            QMessageBox.critical(self, "学习曲线失败", str(e))
            self.info.emit(f"学习曲线失败：{e}")

    # ---------------- 模型 I/O ----------------
    def _save_model(self):
        if self.model is None:
            QMessageBox.information(self, "提示", "请先训练模型。")
            return
        os.makedirs("data/models", exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "data/models/model.pkl", "Pickle (*.pkl)")
        if not path:
            return
        try:
            with open(path, "wb") as f:
                pickle.dump({"model": self.model, "classes": self.classes_}, f)
            self.info.emit(f"模型已保存：{path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            self.info.emit(f"保存失败：{e}")

    def _load_model(self):
        if self.X is None or self.y is None:
            QMessageBox.information(self, "提示", "请先导入CSV或生成演示数据。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "data/models", "Pickle (*.pkl)")
        if not path:
            return
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.model = obj.get("model", None)
            self.classes_ = obj.get("classes", self.classes_)
            if self.model is None:
                raise ValueError("模型文件无效。")

            # 若已有最近一次划分，则沿用比例；否则重新划分
            test_size = float(self.split_spin.value())
            Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y, random_state=0)
            self._last_split = (Xtr, Xte, ytr, yte)

            ypred = self.model.predict(Xte)
            report = classification_report(yte, ypred, target_names=[str(c) for c in self.classes_])
            cm = confusion_matrix(yte, ypred)
            self.txt_report.setPlainText("—— 加载模型评估 ——\n" + report)
            self._plot_confusion_matrix(cm, [str(c) for c in self.classes_])
            self._plot_roc(Xte, yte)
            self.info.emit(f"已加载模型并完成评估：{os.path.basename(path)}")

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self.info.emit(f"加载失败：{e}")

    # ---------------- 批量对比实验（可选） ----------------
    def _run_comparison(self):
        try:
            if self.X is None or self.y is None:
                QMessageBox.information(self, "提示", "请先导入CSV或生成演示数据。")
                return

            test_size = float(self.split_spin.value())
            Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y, random_state=0)
            cv = int(self.cv_spin.value())
            metric = self.cmb_cmp_metric.currentText()

            # 按 UI 选择集成算法与其简易网格
            todo = []
            if self.chk_cmp_svm_rbf.isChecked():
                todo.append(("SVM-RBF", SVC(kernel="rbf", probability=True, random_state=0),
                             {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}))
            if self.chk_cmp_svm_lin.isChecked():
                todo.append(("SVM-Linear", SVC(kernel="linear", probability=True, random_state=0),
                             {"C": [0.1, 1, 10]}))
            if self.chk_cmp_knn.isChecked():
                todo.append(("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]}))
            if self.chk_cmp_lr.isChecked():
                todo.append(("LR", LogisticRegression(max_iter=300, random_state=0),
                             {"C": [0.1, 1, 10], "solver": ["lbfgs"]}))
            if self.chk_cmp_rf.isChecked():
                todo.append(("RF", RandomForestClassifier(random_state=0),
                             {"n_estimators": [100, 200], "max_depth": [3, 5, 7]}))

            if not todo:
                QMessageBox.information(self, "提示", "请至少勾选一种算法进行对比。")
                return

            # 统一的特征步骤（按当前UI）
            base_feat = self._build_feature_steps()
            results = []
            self.info.emit(f"开始批量对比：算法数={len(todo)}，评分={metric}")

            for name, clf, g in todo:
                est = Pipeline(base_feat + [("clf", clf)])
                grid = {f"clf__{k}": v for k, v in g.items()}
                gs = GridSearchCV(estimator=est, param_grid=grid, scoring=metric,
                                  cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0),
                                  n_jobs=-1)
                gs.fit(Xtr, ytr)
                best = gs.best_estimator_
                scr_cv = gs.best_score_
                ypred = best.predict(Xte)
                if metric == "accuracy":
                    scr_te = (ypred == yte).mean()
                else:
                    # 简易：macro f1
                    from sklearn.metrics import f1_score
                    scr_te = f1_score(yte, ypred, average="macro")
                results.append((name, scr_cv, scr_te, best))

            # 文本输出
            lines = ["—— 批量对比结果 ——", f"评分指标（CV）：{metric}"]
            for name, scv, ste, _best in results:
                lines.append(f"{name:>12s} | CV={scv:.4f} | Test={ste:.4f}")
            self.txt_report.append("\n" + "\n".join(lines))

            # 图：右下角条形图
            ax = self.canvas.ax_cmp
            ax.clear()
            labels = [r[0] for r in results]
            cv_scores = [r[1] for r in results]
            te_scores = [r[2] for r in results]
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w/2, cv_scores, width=w, label="CV")
            ax.bar(x + w/2, te_scores, width=w, label="Test")
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10)
            ax.set_ylim(0, 1.0)
            ax.set_title("批量对比（更高更好）")
            ax.legend()
            self.canvas.draw()

            self.info.emit("批量对比完成。图表与结果已更新。")

        except Exception as e:
            QMessageBox.critical(self, "对比失败", str(e))
            self.info.emit(f"对比失败：{e}")
