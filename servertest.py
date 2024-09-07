from QKD.Worker import Bob_Server, Server
import time
import sys
import numpy as np
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
        self.server = Bob_Server()
        button = QPushButton("write to alice")
        test = np.array([
            [3, 235, 238, 700, 54],
            [4, 199, 202, 696, 54],
            [4, 195, 186, 717, 55],
            [4, 163, 177, 682, 54],
        ])
        test = np.linspace(0, 100, int(1E8))
        print("Sending {}GB".format(test.itemsize * test.size / 1E9))
        button.clicked.connect(lambda: self.server.send_alice(test))
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
        gui.server.kill()
        sys.exit(a)
