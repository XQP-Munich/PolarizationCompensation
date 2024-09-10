from QKD.Worker import Alice_Client
from Devices.TimeTaggerLmu import TimeTaggerLmu
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
import numpy as np

beamsplitter_to_alice = (2.6 + 2.5) / ((257.3 + 256.4))
beamsplitter_to_bob = (234.4 + 234.5) / ((257.3 + 256.4))
beamsplitter_alice_to_bob = beamsplitter_to_bob / beamsplitter_to_alice

eff_alice_apd = 0.0248

eff_alice_optics = 10**(-0.36 / 10)

rep_rate = 100E6


class Gui(QWidget):

    def __init__(self, eff_alice_optics=eff_alice_optics, *args, **kwargs):
        super(Gui, self).__init__(*args, **kwargs)
        self.eff_alice_optics = eff_alice_optics
        self.client = Alice_Client()

        button = QPushButton("write to server")
        button.clicked.connect(lambda: self.client.send("boks"))
        layout = QVBoxLayout()
        layout.addWidget(button)

        self.tt = TimeTaggerLmu(simulate=True)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout)
        self.timer.start(5000)
        self.setLayout(layout)
        #self.show()
        self.i = 0

    def timeout(self):
        cps = self.tt.get_counts_per_second(4)
        cps = cps / (1 - cps * 77E-9)
        cps_ex_aper = cps / eff_alice_apd * beamsplitter_alice_to_bob * self.eff_alice_optics
        mu = cps_ex_aper / rep_rate
        data = [time.time(), mu]
        print(data)
        self.client.send(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    gui = Gui()
    try:
        a = app.exec_()
    finally:
        gui.client.kill()
        sys.exit()
