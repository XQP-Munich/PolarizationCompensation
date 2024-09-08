import sys
import time
import os
import shutil
import numpy as np
from scipy import signal as sgnl
from scipy import stats as stats
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import toml
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

from PyQt5.QtGui import (
    QFont, )

from QKD.Worker import Bob_Server, Experimentor
from Devices.AliceLmu import AliceLmu
from Devices.TimeTaggerUltra import TimeTaggerUltra

matplotlib.use('QtAgg')

# setup variables

channels = {
    "H": {
        "ch": 1,
        "edge": -1,
        "trigger": -0.5
    },
    "V": {
        "ch": 2,
        "edge": -1,
        "trigger": -0.5
    },
    "P": {
        "ch": 3,
        "edge": -1,
        "trigger": -0.5
    },
    "M": {
        "ch": 4,
        "edge": -1,
        "trigger": -0.5
    },
    "SYNC": {
        "ch": 5,
        "edge": 1,
        "trigger": 0.25
    },
    "CLK": {
        "ch": 6,
        "edge": 1,
        "trigger": 0.25
    }
}
# 50/50
aliceSettings = [
    [3, 235, 238, 700, 54],
    [4, 199, 202, 696, 54],
    [4, 195, 186, 717, 55],
    [4, 163, 177, 682, 54],
]
# 70 / 30
#aliceSettings = [
#    [3, 230, 238, 700, 54],
#    [3, 236, 241, 696, 54],
#    [4, 189, 184, 717, 55],
#    [4, 159, 173, 682, 58],
#]
# new
#aliceSettings = [
#    [3, 251, 238, 700, 55],
#    [4, 197, 198, 696, 54],
#    [4, 188, 184, 717, 55],
#    [4, 173, 187, 682, 54],
#]

aliceSettingsFile = "aliceConfig.toml"
if os.path.isfile(aliceSettingsFile):
    with open(aliceSettingsFile, "r") as f:
        aliceSettingsContent = f.read()
    aliceSettingsContent = toml.loads(aliceSettingsContent)
    settings = ["bias", "modSig", "modDec", "delayA", "delayB"]
    aliceSettings = []
    for i in range(1, 5):
        t = aliceSettingsContent["laserConfig"]["channel{}".format(i)]
        chan_settings = []
        for setting in settings:
            if setting == "delayB":
                chan_settings.append(t[setting] - chan_settings[-1])
            else:
                chan_settings.append(t[setting])
        aliceSettings.append(chan_settings)

symbol_map = {k: v["ch"] for k, v in channels.items() if k != "CLK"}
inv_symbol_map = {v: k for k, v in symbol_map.items()}

state_map = {"H": 0, "V": 1, "P": 2, "M": 3, "h": 4, "v": 5, "p": 6, "m": 7}
inv_state_map = {v: k for k, v in state_map.items()}

filter = 1.32 / 100
filter = 0.0001  #40dBrep
bob_eff = filter * 25.3852 / 100
alice_trans = 0.917 / 0.9954
rep_rate = 100E6
chan_offset = [12000, 12071, 10000 - 4515, 13070, 12000]
dt = 10000
meas_time = 0
key_length = 0


class MplCanvas(FigureCanvas):

    def __init__(self, pol=0, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.pol = pol
        self.resetData()
        super(MplCanvas, self).__init__(self.fig)

    def resetData(self):
        self.xdata = None
        self.ydata = None


class CustomDialog(QDialog):

    def __init__(
            self,
            parent=None,
            settings=[100E6, -0.356, -5.95, True, False, False, 0.33, -18.79]):
        super().__init__(parent)

        self.setWindowTitle("Settings")
        self.settings = settings

        self.settings_layout = QVBoxLayout()

        self.alice_settings_box = QGroupBox("Alice")
        self.alice_settings_box_layout = QGridLayout()

        self.alice_host_label = QLabel("Host:")
        self.alice_settings_box_layout.addWidget(self.alice_host_label, 0, 0)
        self.alice_host_edit = QLineEdit()
        self.alice_settings_box_layout.addWidget(self.alice_host_edit, 0, 1)

        self.alice_rep_label = QLabel("Alice Reprate:")
        self.alice_settings_box_layout.addWidget(self.alice_rep_label, 1, 0)
        self.alice_rep_spin = QDoubleSpinBox()
        self.alice_rep_spin.setRange(1.0, 1000.0)
        self.alice_rep_spin.setSuffix("MHz")
        self.alice_settings_box_layout.addWidget(self.alice_rep_spin, 1, 1)

        self.alice_loss_label = QLabel("Alice Loss:")
        self.alice_settings_box_layout.addWidget(self.alice_loss_label, 2, 0)
        self.alice_loss_spin = QDoubleSpinBox()
        self.alice_loss_spin.setRange(-100.0, 0.0)
        self.alice_loss_spin.setSuffix("dB")
        self.alice_loss_spin.setSingleStep(0.1)
        self.alice_settings_box_layout.addWidget(self.alice_loss_spin, 2, 1)

        self.alice_settings_box.setLayout(self.alice_settings_box_layout)
        self.settings_layout.addWidget(self.alice_settings_box)

        self.bob_settings_box = QGroupBox("Bob")
        self.bob_settings_box_layout = QGridLayout()

        self.bob_loss_label = QLabel("Bob Loss:")
        self.bob_settings_box_layout.addWidget(self.bob_loss_label, 0, 0)

        self.bob_loss_spin = QDoubleSpinBox()
        self.bob_loss_spin.setRange(-100.0, 0.0)
        self.bob_loss_spin.setSuffix("dB")
        self.bob_loss_spin.setSingleStep(0.1)
        self.bob_settings_box_layout.addWidget(self.bob_loss_spin, 0, 1)

        self.bob_settings_box.setLayout(self.bob_settings_box_layout)
        self.settings_layout.addWidget(self.bob_settings_box)

        self.eval_settings_box = QGroupBox("Evaluation")
        self.eval_settings_box_layout = QGridLayout()
        self.eval_mu_radio = QRadioButton("Plot mu using fixed Loss:")
        self.eval_mu_radio.setChecked(True)
        self.eval_mu_radio.toggled.connect(self.eval_radio_changed)
        self.eval_settings_box_layout.addWidget(self.eval_mu_radio, 0, 0)

        self.eval_floss_spin = QDoubleSpinBox()
        self.eval_floss_spin.setRange(-100.0, 0.0)
        self.eval_floss_spin.setSuffix("dB")
        self.eval_floss_spin.setSingleStep(0.1)
        self.eval_settings_box_layout.addWidget(self.eval_floss_spin, 0, 1)

        self.eval_floss_radio = QRadioButton("Plot Loss using fixed Mu:")
        self.eval_floss_radio.toggled.connect(self.eval_radio_changed)
        self.eval_settings_box_layout.addWidget(self.eval_floss_radio, 1, 0)

        self.eval_fmu_spin = QDoubleSpinBox()
        self.eval_fmu_spin.setRange(0.0, 10.0)
        self.eval_fmu_spin.setSingleStep(0.01)
        self.eval_fmu_spin.setDecimals(3)
        self.eval_settings_box_layout.addWidget(self.eval_fmu_spin, 1, 1)

        self.eval_lloss_radio = QRadioButton("Plot Loss using live Mu")
        self.eval_lloss_radio.toggled.connect(self.eval_radio_changed)
        self.eval_settings_box_layout.addWidget(self.eval_lloss_radio, 2, 0, 1,
                                                2)

        self.eval_settings_box.setLayout(self.eval_settings_box_layout)
        self.settings_layout.addWidget(self.eval_settings_box)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.settings_layout.addWidget(self.buttonBox)

        self.setLayout(self.settings_layout)
        self.eval_radio_changed()
        self.accepted.connect(self.update_settings)
        self.rejected.connect(self.reset_settings)
        self.reset_settings()

    def eval_radio_changed(self):
        if self.eval_mu_radio.isChecked():
            self.eval_floss_spin.setEnabled(True)
            self.eval_fmu_spin.setEnabled(False)
        elif self.eval_floss_radio.isChecked():
            self.eval_floss_spin.setEnabled(False)
            self.eval_fmu_spin.setEnabled(True)
        elif self.eval_lloss_radio.isChecked():
            self.eval_fmu_spin.setEnabled(False)
            self.eval_floss_spin.setEnabled(False)

    def update_settings(self):
        self.settings = [
            self.alice_rep_spin.value() * 1E6,
            self.alice_loss_spin.value(),
            self.bob_loss_spin.value(),
            self.eval_mu_radio.isChecked(),
            self.eval_floss_radio.isChecked(),
            self.eval_lloss_radio.isChecked(),
            self.eval_fmu_spin.value(),
            self.eval_floss_spin.value()
        ]

    def reset_settings(self):
        self.alice_rep_spin.setValue(self.settings[0] / 1E6)
        self.alice_loss_spin.setValue(self.settings[1])
        self.bob_loss_spin.setValue(self.settings[2])
        self.eval_mu_radio.setChecked(self.settings[3])
        self.eval_floss_radio.setChecked(self.settings[4])
        self.eval_lloss_radio.setChecked(self.settings[5])
        self.eval_fmu_spin.setValue(self.settings[6])
        self.eval_floss_spin.setValue(self.settings[7])

    def get_settings(self):
        return self.settings


class Gui(QWidget):
    stop_signal = pyqtSignal()
    pause_signal = pyqtSignal()
    start_signal = pyqtSignal()
    alice_settings_set_signal = pyqtSignal(object)
    eval_settings_changed_signal = pyqtSignal(object)

    def __init__(self,
                 mode,
                 folder,
                 key_file,
                 channels,
                 alice=None,
                 timestamp=None,
                 *args,
                 **kwargs):
        super(Gui, self).__init__(*args, **kwargs)

        self.folder = folder
        self.key_file = key_file
        self.channels = channels
        self.alice = alice
        self.timestamp = timestamp
        self.aliceSettingsSpinBoxes = []
        self.settingsTimer = QTimer()
        self.settingsTimer.setSingleShot(True)
        self.settingsTimer.timeout.connect(self.updateAliceSettings)
        self.aliceSettings = aliceSettings
        self.frame = 0
        self.mode = mode
        self.initUI()
        self.initWorkers()

    def initUI(self):
        self.canvases = []
        for i in range(8):
            self.canvases.append(
                MplCanvas(pol=i % 4, width=4, height=5, dpi=100))

        self.settings = CustomDialog()
        # Buttons:        self.btn_open = QPushButton('Open')

        # Alice settings

        # GUI title, size, etc...
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle("Live Analysis")
        self.gridlayout = QGridLayout()

        self.btn_settings = QPushButton('Settings')
        self.btn_open = QPushButton('Open')

        self.btn_widget = QWidget()
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.btn_settings)
        self.btn_layout.addWidget(self.btn_open)
        self.btn_widget.setLayout(self.btn_layout)
        self.gridlayout.addWidget(self.btn_widget, 0, 0)

        self.set_widget = QWidget()
        self.set_layout = QHBoxLayout()
        self.set_widget.setLayout(self.set_layout)

        measLabel = QLabel("Measurement Time")
        measLabel.setAlignment(Qt.AlignCenter)
        self.measSpinBox = QSpinBox()
        self.measSpinBox.setAlignment(Qt.AlignCenter)
        self.measSpinBox.setRange(1, 60 * 60)
        self.measSpinBox.setValue(5)
        self.measSpinBox.valueChanged.connect(self.settingsChanged)
        self.set_layout.addWidget(measLabel)
        self.set_layout.addWidget(self.measSpinBox)

        self.gridlayout.addWidget(self.set_widget, 0, 1)

        self.check_save_raw = QCheckBox("Save raw Data")
        self.check_save_raw.stateChanged.connect(self.updateEvaluationSettings)
        self.check_save_sifted = QCheckBox("Save sifted Data")
        self.check_save_sifted.stateChanged.connect(
            self.updateEvaluationSettings)
        self.check_time_filter = QCheckBox("Time Filtering")
        self.check_time_filter.setChecked(True)
        self.check_time_filter.stateChanged.connect(
            self.updateEvaluationSettings)

        self.cb_widget = QWidget()
        self.cb_layout = QHBoxLayout()
        self.cb_layout.addWidget(self.check_time_filter)
        self.cb_layout.addWidget(self.check_save_raw)
        self.cb_layout.addWidget(self.check_save_sifted)
        self.cb_widget.setLayout(self.cb_layout)
        self.gridlayout.addWidget(self.cb_widget, 0, 2)

        self.set_widget = QWidget()
        self.set_layout = QHBoxLayout()
        self.btn_set_Alice = QPushButton('Set Alice')
        self.btn_set_Alice.clicked.connect(self.setAliceSettings)
        self.btn_set_Key = QPushButton('Set Key')
        self.aliceConnectionChanged(False)

        self.set_layout.addWidget(self.btn_set_Key)
        self.set_layout.addWidget(self.btn_set_Alice)
        self.set_widget.setLayout(self.set_layout)
        self.gridlayout.addWidget(self.set_widget, 0, 3)

        #self.layout.addWidget(self.btn_open, 1, 3)
        for i in range(4):
            laserGroup = QGroupBox("{} Polarization".format(inv_state_map[i]))
            #laserGroup.setFlat(True)
            laserLayout = QGridLayout()
            aliceSettingsGroup = QGroupBox("Laser Setting")
            aliceSettingsLayout = QGridLayout()
            settings = [["Bias", (0, 20), 0],
                        ["Signal\nmodulation", (0, 255), 0],
                        ["Decoy\nmodulation", (0, 255), 0],
                        ["Pulse\ntiming", (0, 1023), 0],
                        ["Pulse\nwidth", (0, 1023), 0]]
            self.aliceSettingsSpinBoxes.append([])
            for j in range(5):
                settingsLabel = QLabel(settings[j][0])
                settingsLabel.setAlignment(Qt.AlignCenter)
                settingsSpinBox = QSpinBox()
                self.aliceSettingsSpinBoxes[-1].append(settingsSpinBox)
                settingsSpinBox.setAlignment(Qt.AlignCenter)
                if j != 4:
                    settingsSpinBox.setRange(*settings[j][1])
                else:
                    settingsSpinBox.setRange(0, 1023 - aliceSettings[i][j - 1])
                settingsSpinBox.setValue(aliceSettings[i][j])
                if False:  #self.mode == 0:
                    settingsSpinBox.setEnabled(False)
                else:
                    settingsSpinBox.valueChanged.connect(self.settingsChanged)
                aliceSettingsLayout.addWidget(settingsLabel, 0, j)
                aliceSettingsLayout.addWidget(settingsSpinBox, 1, j)
            aliceSettingsGroup.setLayout(aliceSettingsLayout)

            laserLayout.addWidget(self.canvases[i], 0, 0, 1, 1)
            laserLayout.addWidget(self.canvases[i + 4], 1, 0, 1, 1)
            laserLayout.addWidget(aliceSettingsGroup, 2, 0)
            laserGroup.setLayout(laserLayout)

            self.gridlayout.addWidget(laserGroup, 1, i, 1, 1)

        statisticsGroup = QGroupBox("Stats")
        statisticsLayout = QHBoxLayout()
        settings = [["Frame", (0, 255), 0], ["QBER", (0, 255), 0],
                    ["Sifted Key Rate", (0, 1023), 0],
                    ["Sync Offset", (0, 1023), 0], ["Offsets", (0, 1023), 0],
                    ["Key match", (0, 1023), 0]]
        self.statTexts = []
        for j in range(len(settings)):
            statisticsLabel = QLabel(settings[j][0])
            statisticsLabel.setAlignment(Qt.AlignRight)
            cfont = QFont()
            #cfont.setPointSize(18)
            statisticsLabel.setFont(cfont)
            statisticsText = QLabel("0.53")
            self.statTexts.append(statisticsText)
            statisticsText.setAlignment(Qt.AlignLeft)
            statisticsLayout.addWidget(statisticsLabel)
            statisticsLayout.addWidget(statisticsText)
        statisticsGroup.setLayout(statisticsLayout)

        self.gridlayout.addWidget(statisticsGroup, 3, 0, 1, 4)

        # for i, canvas in enumerate(self.canvases):
        #     self.layout.addWidget(canvas, 1 + i, 0)
        self.setLayout(self.gridlayout)

        geometry = app.desktop().availableGeometry()

        self.setGeometry(geometry)  # Thread:

        # Start Button action:

        # Stop Button action:
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_open.clicked.connect(self.getfile)

        self.show()

    def updateLaser(self, data):
        pass

    def updateStats(self, data):
        self.frame = int(data[0])
        for text, data in zip(self.statTexts, data):
            if data is not None:
                text.setText(data)

    def saveSettings(self, settings):
        with open(self.folder + "aliceSettings.csv", "a") as f:
            f.writelines("{}\t{}\n".format(self.frame, settings))

    def loadSettings(self, settings):
        path = self.folder + "aliceSettings.csv"
        if os.path.isfile(path):
            with open(path, "r") as f:
                for line in f.readlines:
                    frame, settings = line[:-1].split("\t")

    def aliceConnectionChanged(self, connected):
        self.btn_set_Alice.setEnabled(connected)
        self.btn_set_Key.setEnabled(connected)

    def settingsChanged(self):
        print("settings changed")
        if self.settingsTimer.isActive():
            self.settingsTimer.stop()
        self.settingsTimer.start(500)

    def updateAliceSettings(self):
        self.aliceSettings = []
        for p in self.aliceSettingsSpinBoxes:
            self.aliceSettings.append([])
            for i, sbox in enumerate(p):
                if i == 4:
                    sbox.setRange(0, 1023 - p[i - 1].value())
                self.aliceSettings[-1].append(sbox.value())

    def setAliceSettings(self):
        self.alice_settings_set_signal.emit(self.aliceSettings)
        self.saveSettings(self.aliceSettings)

    def updateEvaluationSettings(self):
        self.eval_settings_changed_signal.emit([
            self.check_time_filter.isChecked(),
            self.check_save_raw.isChecked(),
            self.check_save_sifted.isChecked()
        ])

    def initWorkers(self):
        self.threadpool = QThreadPool.globalInstance()
        self.server = Bob_Server()
        self.server.alice_connected.connect(self.aliceConnectionChanged)
        self.stop_signal.connect(self.server.kill)
        experimentor = Experimentor(self.canvases,
                                    self.mode,
                                    self.folder,
                                    self.channels,
                                    key_file=self.key_file,
                                    alice=self.alice,
                                    timestamp=self.timestamp)
        self.stop_signal.connect(experimentor.kill)
        self.pause_signal.connect(experimentor.pause)
        self.start_signal.connect(experimentor.resume)
        self.btn_set_Key.clicked.connect(experimentor.send_key)
        self.alice_settings_set_signal.connect(experimentor.set_aliceSettings)
        self.measSpinBox.valueChanged.connect(
            lambda: experimentor.set_meas_time(self.measSpinBox.value()))
        self.eval_settings_changed_signal.connect(
            experimentor.update_eval_settings)
        experimentor.signals.new_data.connect(self.updateStats)
        self.server.alice_new_data.connect(experimentor.handle_alice_counts)
        self.settings.accepted.connect(
            lambda: experimentor.handle_settings_change(self.settings.
                                                        get_settings()))
        experimentor.handle_settings_change(self.settings.get_settings())
        self.threadpool.start(experimentor)

    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './', "*")
        print(fname)

    def start_threads(self):
        print("resume threads")
        self.start_signal.emit()

    # When stop_btn is clicked this runs. Terminates the self.plotter and the thread.
    def stop_threads(self):
        print("stopping threads")
        self.stop_signal.emit()

    def pause_threads(self):
        print("pausing threads")
        self.pause_signal.emit()  # emit the finished signal on stop

    def open_settings(self):
        self.settings.open()


style = '''
QGroupBox {

    border: 1px solid #76797C;  /* <-----  1px solid #76797C */

    border-radius: 2px;
    margin-top: 20px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top;
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 10px;
}
'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(style)
    key_file = ""
    folder = ""
    mode = 0
    alice = None
    timestamp = None
    if len(sys.argv) > 1:
        file = sys.argv[1]
        if not os.path.exists(file):
            print("Cannot open {}".format(file))
            sys.exit()

        if os.path.isdir(file):
            folder = file + "/"
            for f in os.listdir(file):
                if f.endswith(".txt"):
                    key_file = folder + f
            mode = 0
        else:
            filename, file_extension = os.path.splitext(os.path.basename(file))
            if file_extension == ".txt":
                folder = "./data/measure_qber_" + time.strftime(
                    "%Y%m%d-%H%M%S") + "/"
                key_file = folder + filename + file_extension
                os.makedirs(folder)
                shutil.copyfile(file, key_file)
                mode = 1
    if mode == 1:
        alice = AliceLmu("t14s")
        timestamp = TimeTaggerUltra(channels)
    gui = Gui(mode,
              folder,
              key_file,
              channels,
              alice=alice,
              timestamp=timestamp)
    a = 0
    try:
        a = app.exec_()
    finally:
        gui.stop_threads()
        sys.exit(a)
