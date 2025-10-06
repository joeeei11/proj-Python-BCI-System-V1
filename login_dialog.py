# -*- coding: utf-8 -*-
# login_dialog.py
#
# 说明：
# - 固定账号/密码： admin / 123456
# - 不依赖 bcrypt 或数据库，零外部依赖
# - 保持与主程序接口一致（MainWindow 从 dlg.username_edit 读取用户名）
# - 集成系统日志（log_module），记录登录成功/失败

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QHBoxLayout, QCheckBox
)

# 日志模块（确保项目根目录有 log_module.py）
try:
    from log_module import log_module
except Exception:
    class _DummyLogger:
        def log_info(self, m): pass
        def log_warning(self, m): pass
        def log_error(self, m): pass
        def log_debug(self, m): pass
    log_module = _DummyLogger()

# 固定用户表（如需扩展，可在此新增用户）
VALID_USERS = {
    "admin": "123456"
}

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户登录")
        self.setMinimumWidth(380)
        self.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))

        # —— 表单控件 ——
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("请输入用户名（admin）")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("请输入密码（123456）")
        self.password_edit.setEchoMode(QLineEdit.Password)

        # 显示/隐藏密码
        self.show_pwd = QCheckBox("显示密码")
        self.show_pwd.toggled.connect(
            lambda on: self.password_edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password)
        )

        self.msg_label = QLabel("")   # 提示信息
        self.msg_label.setAlignment(Qt.AlignCenter)

        self.btn_login = QPushButton("登录")
        self.btn_cancel = QPushButton("取消")

        # —— 布局 ——
        form = QFormLayout()
        form.addRow("用户名：", self.username_edit)
        form.addRow("密码：", self.password_edit)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_login)
        btns.addWidget(self.btn_cancel)

        root = QVBoxLayout()
        root.addLayout(form)
        root.addWidget(self.show_pwd, 0, Qt.AlignLeft)
        root.addSpacing(6)
        root.addWidget(self.msg_label)
        root.addSpacing(8)
        root.addLayout(btns)
        self.setLayout(root)

        # —— 样式（苹果风格，圆角+黑白灰+蓝强调） ——
        self.setStyleSheet("""
            QDialog { background: #FFFFFF; }
            QLabel { color:#323232; }
            QLineEdit {
                border:1px solid #D0D0D0; border-radius:10px; padding:8px 10px;
                background:#F7F7F7; color:#000;
            }
            QLineEdit:focus { border:1px solid #007AFF; background:#FFFFFF; }
            QPushButton {
                background:#007AFF; color:#FFF; border:none; border-radius:10px;
                padding:10px 18px; min-width:100px; font-weight:bold;
            }
            QPushButton:hover { background:#1A84FF; }
            QPushButton:pressed { background:#0062CC; }
            QCheckBox { color:#323232; }
        """)

        # —— 信号 ——
        self.btn_login.clicked.connect(self._on_login)
        self.btn_cancel.clicked.connect(self.reject)
        # 回车键直接登录
        self.username_edit.returnPressed.connect(self._on_login)
        self.password_edit.returnPressed.connect(self._on_login)

        log_module.log_info("登录界面已打开")

    def _on_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text()

        # 空检查
        if not username:
            self._set_error("用户名不能为空")
            log_module.log_warning("登录失败：用户名为空")
            return
        if not password:
            self._set_error("密码不能为空")
            log_module.log_warning("登录失败：密码为空")
            return

        # 校验
        if username not in VALID_USERS:
            self._set_error("用户名不存在")
            log_module.log_error(f"登录失败：用户名不存在 - {username}")
            return

        if VALID_USERS[username] != password:
            self._set_error("密码错误")
            log_module.log_error(f"登录失败：密码错误 - {username}")
            return

        # 成功
        self._set_ok("登录成功")
        log_module.log_info(f"用户登录成功 - {username}")
        self.accept()

    def _set_error(self, text: str):
        self.msg_label.setText(text)
        self.msg_label.setStyleSheet("color:#FF3B30; font-weight:bold;")

    def _set_ok(self, text: str):
        self.msg_label.setText(text)
        self.msg_label.setStyleSheet("color:#4CD964; font-weight:bold;")
