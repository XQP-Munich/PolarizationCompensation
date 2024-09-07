from QKD.Worker import Alice_Client, Client
import time

import sys
from PyQt5.QtWidgets import (
    QRadioButton,
    QWidget,
    QPushButton,
    QApplication,
    QGridLayout,
    QFileDialog,
    QLabel,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTextEdit,
    QDialog,
    QDialogButtonBox,
)
from PyQt5.QtCore import (
    QReadWriteLock,
    Qt,
    QThread,
    QObject,
    pyqtSignal,
    QThreadPool,
    QRunnable,
    pyqtSlot,
    QMutex,
    QTimer,
)


class Gui(QWidget):

    def __init__(self, *args, **kwargs):
        super(Gui, self).__init__(*args, **kwargs)
        self.client = Alice_Client()
        button = QPushButton("write to server")
        button.clicked.connect(lambda: self.client.send("boks"))
        layout = QVBoxLayout()
        layout.addWidget(button)
        self.setLayout(layout)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    gui = Gui()
    try:
        a = app.exec_()
    finally:
        gui.client.kill()
        sys.exit()
