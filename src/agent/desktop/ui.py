"""Native, single-composer desktop chat presentation."""
import queue

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QTextBrowser, QVBoxLayout, QWidget,
)

STYLESHEET = """
QWidget { background: #faf9f6; color: #30312e; font: 14px 'Segoe UI'; }
QLabel { background: transparent; }
QFrame#sidebar { background: #eeede8; }
QLabel#brand { font-size: 18px; font-weight: 600; }
QLabel#hero { font-family: Georgia; font-size: 32px; }
QLabel#muted { color: #77796f; font-size: 12px; }
QLabel#status { color: #77796f; font-size: 12px; }
QPushButton { border: 0; border-radius: 8px; padding: 10px 14px; background: transparent; text-align: left; }
QPushButton:hover { background: #e5e3dc; }
QPushButton:disabled { color: #a8a79f; }
QPushButton#send { background: #34372f; color: white; text-align: center; font-size: 20px; }
QPushButton#send:disabled { background: #cecec6; }
QFrame#composer { background: white; border: 1px solid #dedcd3; border-radius: 18px; }
QPlainTextEdit { background: transparent; border: 0; font-size: 15px; padding: 6px; }
QTextBrowser { background: transparent; border: 0; font-size: 15px; }
QFrame#userMessage { background: #efeee8; border-radius: 14px; }
QFrame#assistantMessage { background: transparent; }
QFrame#plan { background: #f0efe9; border: 1px solid #dedcd3; border-radius: 12px; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: transparent; width: 7px; }
QScrollBar::handle:vertical { background: #d3d1c8; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


class Composer(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            self.submitted.emit()
        else:
            super().keyPressEvent(event)


class MessageBody(QTextBrowser):
    """Markdown document sized to its contents, preserving code indentation."""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.document().setDefaultStyleSheet('pre { background-color: #eeede7; font-family: Consolas; } code { font-family: Consolas; }')
        self.setMarkdown(text)
        self.document().documentLayout().documentSizeChanged.connect(self.fit_height)

    def fit_height(self, *_):
        self.setFixedHeight(max(36, int(self.document().size().height()) + 12))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_height()


class ChatWindow(QWidget):
    def __init__(self, presenter):
        super().__init__()
        self.presenter = presenter
        self.events = queue.Queue()
        self.task_id = None
        self.busy = False
        self.setWindowTitle('Personal AI Agent')
        self.setStyleSheet(STYLESHEET)
        self.resize(1120, 800)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.sidebar = QFrame(objectName='sidebar')
        self.sidebar.setFixedWidth(230)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(20, 28, 20, 22)
        side.addWidget(QLabel('Personal AI Agent', objectName='brand'))
        side.addSpacing(24)
        self.new_button = QPushButton('+  Clear conversation')
        self.new_button.clicked.connect(self.clear_conversation)
        side.addWidget(self.new_button)
        side.addStretch()
        side.addWidget(QLabel('WORKSPACE', objectName='muted'))
        workspace = QLabel(str(presenter.runtime.workspace))
        workspace.setWordWrap(True)
        workspace.setTextFormat(Qt.PlainText)
        side.addWidget(workspace)
        outer.addWidget(self.sidebar)
        center = QVBoxLayout()
        center.setContentsMargins(24, 18, 24, 18)
        header = QHBoxLayout()
        toggle = QPushButton('☰')
        toggle.setAccessibleName('Toggle sidebar')
        toggle.clicked.connect(lambda: self.sidebar.setVisible(not self.sidebar.isVisible()))
        header.addWidget(toggle)
        header.addStretch()
        self.status = QLabel('Ready', objectName='status')
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        header.addWidget(self.status)
        center.addLayout(header)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        content = QWidget()
        self.messages = QVBoxLayout(content)
        self.messages.setContentsMargins(12, 24, 12, 24)
        self.messages.setSpacing(24)
        self.hero = QLabel('What would you like to do?', objectName='hero')
        self.hero.setWordWrap(True)
        self.hero.setAlignment(Qt.AlignCenter)
        self.messages.addStretch()
        self.messages.addWidget(self.hero)
        self.messages.addStretch()
        self.scroll.setWidget(content)
        center.addWidget(self.scroll, 1)
        self.plan = QLabel()
        self.plan.setTextFormat(Qt.PlainText)
        self.plan.setWordWrap(True)
        self.plan.hide()
        center.addWidget(self.plan)
        composer = QFrame(objectName='composer')
        composer_layout = QVBoxLayout(composer)
        self.box = Composer()
        self.box.setPlaceholderText('Ask anything, or describe a task…')
        self.box.setFixedHeight(82)
        self.box.submitted.connect(self.submit)
        composer_layout.addWidget(self.box)
        footer = QHBoxLayout()
        footer.addWidget(QLabel('Enter to send · Shift+Enter for a new line', objectName='muted'))
        footer.addStretch()
        self.send = QPushButton('↑', objectName='send')
        self.send.setAccessibleName('Send message')
        self.send.setFixedSize(40, 40)
        self.send.clicked.connect(self.submit)
        footer.addWidget(self.send)
        composer_layout.addLayout(footer)
        center.addWidget(composer)
        outer.addLayout(center, 1)
        presenter.on_message = lambda value: self.events.put(('message', value))
        presenter.on_status = lambda value: self.events.put(('status', value))
        presenter.on_task = lambda value: self.events.put(('task', value))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.drain)
        self.timer.start(60)

    def clear_conversation(self):
        if self.busy:
            return
        while self.messages.count():
            item = self.messages.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.plan.hide()
        self.task_id = None

    def add_message(self, value):
        if self.hero is not None:
            self.clear_conversation_items()
            self.hero = None
        role, separator, text = value.partition(': ')
        if not separator:
            role, text = 'Agent', value
        card = QFrame(objectName='userMessage' if role == 'You' else 'assistantMessage')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(QLabel('You' if role == 'You' else 'Personal AI Agent', objectName='muted'))
        body = MessageBody(text)
        layout.addWidget(body)
        copy = QPushButton('Copy')
        copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        layout.addWidget(copy, 0, Qt.AlignRight)
        self.messages.addWidget(card)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def clear_conversation_items(self):
        while self.messages.count():
            item = self.messages.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def submit(self):
        text = self.box.toPlainText().strip()
        if not text or self.busy:
            return
        self.busy = True
        self.send.setEnabled(False)
        self.new_button.setEnabled(False)
        self.box.clear()
        try:
            future = self.presenter.submit(text)
            future.add_done_callback(lambda done: self.events.put(('done', done)))
        except Exception:
            self.events.put(('done', None))
            self.status.setText('Could not send. Please try again.')

    def drain(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'message':
                self.add_message(value)
            elif kind == 'status':
                self.status.setText(value)
            elif kind == 'task':
                self.task_id = value.task_id
                self.plan.setText(f'Task: {value.state.value}\n{value.error or "Plan preview is not yet available in this interface. Execution is disabled here."}')
                self.plan.show()
            elif kind == 'done':
                self.busy = False
                self.send.setEnabled(True)
                self.new_button.setEnabled(True)
                if value is not None and value.exception() is not None:
                    self.status.setText('Request failed. Please try again.')

    def closeEvent(self, event):
        self.presenter.close()
        super().closeEvent(event)


def launch(presenter) -> int:
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(presenter)
    window.show()
    return app.exec()
