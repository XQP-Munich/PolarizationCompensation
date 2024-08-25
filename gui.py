import sys
import random
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QGridLayout,
    QFileDialog,
    QLabel,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
)
from PyQt5.QtCore import (
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

import numpy as np
import time

import time, sys, os, shutil
import numpy as np
from scipy import signal as sgnl
from scipy import stats as stats
import matplotlib.pyplot as plt
import matplotlib
import time
from PIL import Image
import gc
from Devices.TimeTaggerUltra import TimeTaggerUltra, TimeTaggerUltraVirtual
from Devices.AliceLmu import AliceLmu

matplotlib.use('QtAgg')
#matplotlib.use("qt5agg")

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

aliceSettings = [
    [3, 236, 211, 100, 54],
    [3, 240, 235, 96, 54],
    [4, 193, 173, 117, 55],
    [3, 196, 192, 82, 53],
]
symbol_map = {k: v["ch"] for k, v in channels.items() if k != "CLK"}
inv_symbol_map = {v: k for k, v in symbol_map.items()}

state_map = {"H": 0, "V": 1, "P": 2, "M": 3, "h": 4, "v": 5, "p": 6, "m": 7}
inv_state_map = {v: k for k, v in state_map.items()}

filter = 1.32 / 100
bob_eff = filter * 25.3852 / 100
alice_trans = 0.917
rep_rate = 100E6
chan_offset = [2000, 2071, -4515, 3070, 2000]
dt = 10000
meas_time = 0
key_length = 0


def save_ts(ts, path):
    data = np.bitwise_or(ts[0], np.left_shift(ts[1], 4))
    data.tofile(path)


def save_sifted(sifted, path):
    print("saving {} events".format(len(sifted[0])))
    pol = []
    for chan in sifted[0]:
        if chan == 1:
            pol.append("H")
        if chan == 2:
            pol.append("V")
        if chan == 3:
            pol.append("P")
        if chan == 4:
            pol.append("M")
    with open(path, "w") as f:
        for i in range(len(sifted[1])):
            f.writelines("{} {}\n".format(pol[i], sifted[1][i]))


def load_ts(path):
    data = np.fromfile(path, dtype=np.uint64)
    return np.array([np.bitwise_and(data, 15), np.right_shift(data, 4)])


def guess_key(bins):
    key = []
    qbers = []
    num_sifted_det = []
    for count in bins:
        diff1 = count[0] - count[1]
        diff2 = count[2] - count[3]
        diff3 = abs(diff1) - abs(diff2)
        val = 4
        qber = 0.5
        if diff3 > 0:
            basis_counts = count[0] + count[1]
            if basis_counts > 0:
                if diff1 > 0:
                    val = 0
                    qber = count[1] / basis_counts
                else:
                    val = 1
                    qber = count[0] / basis_counts
        else:
            basis_counts = count[2] + count[3]
            if basis_counts > 0:
                if diff2 > 0:
                    val = 2
                    qber = count[3] / basis_counts
                else:
                    val = 3
                    qber = count[2] / basis_counts
        num_sifted_det.append(basis_counts)
        key.append(val)
        qbers.append(qber)

    return np.array(key), np.array(qbers), np.array(num_sifted_det)


def find_filter(phases, height=0.8):
    bin_width = 1000
    filter_min = []
    filter_max = []
    shifts = []
    target_shifts = [0.4, 0.4, 0.4, 0.4, 0.1]
    heights = [height] * 4 + [0.5]
    for i in range(5):
        bins = np.histogram(phases[1, phases[0] == i + 1],
                            bins=bin_width,
                            range=(0, dt))[0]
        peaks, _ = sgnl.find_peaks(bins, prominence=max(bins) / 2)
        shift = int(bin_width * target_shifts[i] - peaks[0])
        shifts.append(int(shift * dt / bin_width))
        bins = np.roll(bins, shift)
        peaks += shift
        _, _, f_min, f_max = sgnl.peak_widths(bins,
                                              peaks,
                                              rel_height=heights[i],
                                              wlen=bin_width)
        filter_min.append(f_min[0] * dt / bin_width)
        filter_max.append(f_max[0] * dt / bin_width)
    return np.array(shifts), np.array([
        np.mean(filter_min[:4]),
        np.mean(filter_max[:4]), filter_min[4], filter_max[4]
    ])


def evaluate_tags(
    data,
    channels,
    offsets,
    sent_key=None,
    key_length=None,
    sync_offset=None,
    filters=None,
    verbose=True,
):
    sync_data = data[1][data[0] == channels["SYNC"]["ch"]]
    sync_phase = 0
    if sent_key is not None:
        key_length = len(sent_key)
    if sync_data.size != 0:
        key_length_sync = int(np.rint(np.mean(np.diff(sync_data)) / dt))
        if key_length is not None and key_length_sync != key_length:
            print(key_length, key_length_sync)
            return []
            #raise Exception(
            #    "Detected Key length does not match expected key length")
        key_length = key_length_sync
        if verbose:
            print("Sync Chan detected")
            print("Key length: {}".format(key_length))
        t0 = sync_data[1]
        data[1] -= t0 - 1000
        data = data[:,
                    np.logical_and(data[1] >= 0, data[1] <= (sync_data[-1] -
                                                             sync_data[1]))]
        sync_data = data[1][data[0] == channels["SYNC"]["ch"]]
        sync_phase = int(np.rint(stats.circmean(sync_data % dt, high=dt)))
        if verbose:
            print("Phase: {}".format(sync_phase))
        meas_time = (data[1][-1] - data[1][0]) * 1E-12
        del sync_data
    else:
        t0 = data[1][1]
        meas_time = (data[1][-1] - t0) * 1E-12

    if verbose:
        print("Got {:.2e} Events in {:.0f}s".format(len(data[0]), meas_time))

    if filters is None:
        temp_data = data.copy()
        for i in range(0, 5):
            temp_data[1, temp_data[0] == i + 1] += offsets[i]
        shifts, filters, = find_filter(temp_data % dt)
        offsets += shifts + sync_phase

    for i in range(0, 5):
        data[1, data[0] == i + 1] += offsets[i] - sync_phase

    del sync_phase

    phases = np.vstack(
        (data % dt, ((data[1] % (dt * key_length)) / dt).astype(int)))
    time_mask = np.logical_or(
        np.logical_and(phases[1] >= filters[0], phases[1] <= filters[1]),
        data[0] == channels["SYNC"]["ch"])
    data = data[:, time_mask]

    sifted_events = data[:, data[0] != 5]
    sifted_events[1] = sifted_events[1] // dt

    bins = np.zeros((key_length, 4))
    pos_in_key = ((data[1] % (dt * key_length)) / dt).astype(int)
    for i in range(0, 4):
        pos_in_key_pol = pos_in_key[data[0] == i + 1]
        unique_indices, counts = np.unique(pos_in_key_pol, return_counts=True)
        for id, cnt in zip(unique_indices, counts):
            bins[id][i] += cnt

    key_guess, qbers, num_sifted_det = guess_key(bins)
    if sent_key is not None and sync_offset is None:
        matching_symbols, sync_offset = compare_key(sent_key, key_guess)
        if verbose:
            if (matching_symbols == key_length):
                print("Keys match with offset: {}".format(sync_offset))
            else:
                print("Keys match with {}/{} symbols, most likely offset: {}".
                      format(matching_symbols, key_length, sync_offset))
    used_key = np.array(key_guess)
    if sent_key is not None and sync_offset is not None:
        sent_key = (sent_key + sent_key)[sync_offset:sync_offset + key_length]
        used_key = np.array(sent_key)

    phases[2] = used_key[phases[2]]

    if verbose:
        print("Mean Qber: {:.2f}% with std of {:.4f}".format(
            np.nanmean(qbers) * 100,
            np.nanstd(qbers) * 100))
        print("Max Qber: {:.2f}% at {}".format(
            np.max(qbers) * 100, bins[np.argmax(qbers)]))
        print("Sifted key Rate: {:.2f}".format(
            np.sum(num_sifted_det) / meas_time))

    mus = []
    for pol in range(8):
        pol_bin = bins[used_key == pol, :]
        mu = pol_bin.sum() / meas_time / bob_eff / rep_rate / (
            len(pol_bin) / key_length) * alice_trans
        mus.append(mu)

    return [
        phases,
        offsets,
        filters,
        sync_offset,
        qbers,
        num_sifted_det,
        meas_time,
        t0 * 1E-12,
        sifted_events,
        mus,
    ]


def get_valid_frames(data, verbose=True):
    valid_timing = 500  #in ps
    sync_indices = np.where(data[0] == channels["SYNC"]["ch"])[0]
    if len(sync_indices) <= 20:
        return []
    sync_diffs = np.diff(data[1][sync_indices], append=0)
    sync_diffs_median = np.median(sync_diffs)
    sync_dropouts = np.where(
        np.logical_or(sync_diffs > (sync_diffs_median + valid_timing),
                      sync_diffs < (sync_diffs_median - valid_timing)))[0]
    if len(sync_dropouts) > 0 and verbose:
        print("{} SYNC dropouts detected:\n{}".format(
            len(sync_dropouts), sync_indices[sync_dropouts]))
    sync_dropouts = np.concatenate(([-1], sync_dropouts))
    valid_frames = []
    start = data[1][sync_indices[0]]
    frame_nr = 0
    for i in range(len(sync_dropouts) - 1):
        beg = sync_indices[sync_dropouts[i] + 1]
        end = sync_indices[sync_dropouts[i + 1]]
        d = data[:, beg:end + 1]
        d[1] += -d[1][0] + int(frame_nr * sync_diffs_median) + start
        frame_nr = (data[1][end] /
                    sync_diffs_median) + 1  # todo get frame_nr from experiment
        valid_frames.append(d)
    if len(sync_dropouts) == 0:
        valid_frames = [data]
    return np.concatenate(valid_frames, axis=1)


def compare_key(sent_key, detected_key):
    temp_key = sent_key + sent_key
    sames = []
    for i in range(len(sent_key)):
        comp_key = temp_key[i:i + len(sent_key)]
        same = 0
        for j in range(len(sent_key)):
            if (detected_key[j] == comp_key[j] % 4):
                same += 1
        sames.append(same)
    return max(sames), np.argmax(sames)


class WorkerSignals(QObject):
    new_stat_data = pyqtSignal(object)
    new_phase_data = pyqtSignal(object)
    new_mu_data = pyqtSignal(object)


class Worker(QRunnable):

    signals = WorkerSignals()

    def __init__(self):
        super().__init__()

        self.is_paused = False
        self.is_running = True

    @pyqtSlot()
    def run(self):
        i = 0
        while self.is_running:
            self.loop(i)
            time.sleep(0.1)
            while self.is_paused and self.is_running:
                print("sleeping")
                QThread.sleep(1)
            i += 1

    def loop(self, frame):
        pass

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def kill(self):
        self.is_running = False


class LinePlotter(Worker):

    def __init__(self, canvas, nr):
        Worker.__init__(self)
        self.nr = nr
        self.canvas = canvas
        self.plot_data = []
        self.mutex = QMutex()
        self.has_new_data = False
        self.xdata = None
        self.ydata = None
        self.frame = 0

    def loop(self, frame):
        if self.has_new_data:
            self.mutex.lock()
            self.has_new_data = False
            mus, = self.plot_data
            self.mutex.unlock()
            self.canvas.axes.cla()
            ax = self.canvas.axes
            start = time.time()
            if self.xdata is None:
                self.xdata = []
                self.ydata = []
            if len(self.xdata) >= 10:
                self.xdata = self.xdata[1:]
                self.ydata = self.ydata[1:]
            self.xdata.append(self.frame)
            self.ydata.append([
                mus[self.nr],
                np.mean([mu for i, mu in enumerate(mus[:4]) if i != self.nr]),
                mus[self.nr + 4] * 10,
                np.mean([mu
                         for i, mu in enumerate(mus[4:]) if i != self.nr]) * 10
            ])
            ax.plot(self.xdata,
                    self.ydata,
                    label=["$\\mu$", "$\\mu_r$", "$\\nu$", "$\\nu_r$"])
            ax.legend(loc="center left")
            ax.hlines([0.51, 0.15],
                      self.xdata[0],
                      self.xdata[-1],
                      linestyles="dashed",
                      colors="gray")

            ax.set_title("$\\mu={:.2f}$\t$\\nu={:.2f}$".format(
                self.ydata[-1][0], self.ydata[-1][2]))

            ax.set_xlabel("Frame")
            ax.set_ylabel("Mean photon number")
            print("plot took {:.2f}s".format(time.time() - start))
            self.canvas.draw()

    def plot_new_data(self, data):
        self.mutex.lock()
        self.plot_data = data
        self.has_new_data = True
        self.frame += 1
        self.mutex.unlock()


class PhasePlotter(Worker):

    def __init__(self, canvas, nr):
        Worker.__init__(self)
        self.nr = nr
        self.canvas = canvas
        self.plot_data = []
        self.mutex = QMutex()
        self.has_new_data = False

    def loop(self, frame):
        if self.has_new_data:
            self.mutex.lock()
            self.has_new_data = False
            hists, bins, filters = self.plot_data
            self.mutex.unlock()
            self.canvas.axes.cla()
            ax = self.canvas.axes
            start = time.time()
            hist_sig = hists[self.nr]
            hist_dec = hists[self.nr + 4] * 40
            hist_other = np.sum([
                hist for i, hist in enumerate(hists[:-1])
                if (i != self.nr and i != self.nr + 4)
            ],
                                axis=0) / 3
            ax.bar(bins[:-1],
                   hist_sig,
                   width=100,
                   alpha=0.4,
                   label=inv_symbol_map[self.nr + 1] + " Signal",
                   color="C{}".format(self.nr))
            ax.bar(bins[:-1],
                   hist_dec,
                   width=100,
                   alpha=0.4,
                   label=inv_symbol_map[self.nr + 1] + " Decoy",
                   color="C{}".format(self.nr + 4))
            ax.bar(bins[:-1],
                   hist_other,
                   width=100,
                   alpha=0.6,
                   label="Rest",
                   color="gray")
            ax.bar(bins[:-1],
                   hists[-1],
                   width=100,
                   alpha=0.4,
                   label="SYNC*",
                   color="C{}".format(5))
            ax.vlines(filters[0], 0, max(hists[-1]))
            ax.vlines(filters[1], 0, max(hists[-1]))
            ax.annotate("Δ  = {:.0f}ps".format((filters[1] - filters[0])),
                        (filters[0] - 100, max(hists[-1]) * 0.9),
                        fontsize="8",
                        ha="right",
                        va="top")
            ax.annotate("FWHM = {:.0f}ps".format((filters[3] - filters[2])),
                        (filters[3] + 100, 20),
                        fontsize="8")
            ax.legend()

            ax.set_xlabel("Detection time in ps")
            ax.set_ylabel("Number of detections")
            print("plot took {:.2f}s".format(time.time() - start))
            self.canvas.draw()

    def plot_new_data(self, data):
        self.mutex.lock()
        self.plot_data = data
        self.has_new_data = True
        self.mutex.unlock()


class Evaluator(Worker):
    new_phase_data = pyqtSignal(object)

    def __init__(self, folder, key_file):
        Worker.__init__(self)
        self.mutex = QMutex()
        self.frame = 0
        self.offsets = chan_offset
        self.folder = folder
        self.key_file = key_file
        self.loop_playback = True
        self.data_files = []
        self.get_data_files()
        self.sent_key = None
        self.sync_offset = None
        self.verbose = True
        if self.key_file != "":
            self.get_key()

    def get_data_files(self):
        for f in os.listdir(self.folder):
            if f.endswith(".bin"):
                self.data_files.append(f)
        print(self.data_files)

    def get_data(self, frame):
        if len(self.data_files) != 0:
            if self.loop_playback:
                frame = frame % len(self.data_files)
            else:
                if frame >= len(self.data_files):
                    time.sleep(1)
                    return [[], []]
            data = load_ts(self.folder + self.data_files[frame])
            return data.view(np.int64)
        else:
            print("Live Measuring not implemented yet")
            return [[], []]

    def get_key(self):
        self.sent_key = []

        with open(self.key_file, "r") as file:
            for char in file.readline()[:-1]:
                self.sent_key.append(state_map[char])

    def loop(self, frame):
        print("Reading in Frame {}".format(frame))
        if self.loop_playback:
            frame = frame % len(self.data_files)
        else:
            if frame >= len(self.data_files):
                time.sleep(1)
                return
        start = time.time()
        data = load_ts(self.folder + self.data_files[frame])
        data = data.view(np.int64)

        if len(data[0]) <= 0:
            print("No Timestamps in frame")
            return
        data = get_valid_frames(data, verbose=False)
        if len(data) <= 0:
            print("No Sync detected in frame")
            return
        et = evaluate_tags(data,
                           channels,
                           self.offsets,
                           sync_offset=self.sync_offset,
                           sent_key=self.sent_key,
                           verbose=self.verbose)
        if len(et) == 0:
            print("Evaluation error, continue")
            return
        phases, self.offsets, self.filters, self.sync_offset = et[:4]
        qbers, num_sifted_det, meas_time, t0 = et[4:8]
        sifted_events, mus = et[8:]
        self.verbose = False
        nmax = []
        hists = []
        bins = []
        phases_pol = phases[:, phases[0] != 5]
        for pol in range(8):
            hist, bins = np.histogram(phases_pol[1, phases_pol[2] == pol],
                                      bins=100,
                                      range=[0, dt])
            hists.append(hist)
            nmax.append(np.max(hist))
        hist, bins = np.histogram(phases[1, phases[0] == 5],
                                  bins=100,
                                  range=[0, dt])
        hists.append(hist)
        nmax.append(np.max(hist))

        hists[-1] = hists[-1] * (np.mean(nmax[:-1]) / nmax[-1])
        self.signals.new_phase_data.emit([hists, bins, self.filters])
        self.signals.new_mu_data.emit([mus])
        self.signals.new_stat_data.emit([
            "{}".format(self.frame),
            "{:.2f}%".format(np.mean(qbers) * 100),
            "{:.2f}Kbit/s".format(np.sum(num_sifted_det) / meas_time / 1000),
            "{:.0f}".format(self.sync_offset),
        ])
        self.frame += 1
        duration = time.time() - start

        print("eval took {:.2f}s".format(duration))
        print("sleep for {}".format(max(0, 2 - duration)))
        time.sleep(max(0, 2 - duration))


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class Gui(QWidget):
    stop_signal = pyqtSignal()
    pause_signal = pyqtSignal()
    start_signal = pyqtSignal()
    settings_changed_signal = pyqtSignal(object)

    def __init__(self, folder, key_file, *args, **kwargs):
        super(Gui, self).__init__(*args, **kwargs)

        self.folder = folder
        self.key_file = key_file
        self.aliceSettingsSpinBoxes = []
        self.settingsTimer = QTimer()
        self.settingsTimer.setSingleShot(True)
        self.settingsTimer.timeout.connect(self.updateSettings)

        self.initUI()

    def initUI(self):
        self.canvases = []
        for i in range(8):
            self.canvases.append(MplCanvas(self, width=4, height=5, dpi=100))

        # Buttons:
        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_open = QPushButton('Open')

        # Alice settings

        # GUI title, size, etc...
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('ThreadTest')
        self.layout = QGridLayout()
        self.layout.addWidget(self.btn_start, 0, 0)
        self.layout.addWidget(self.btn_stop, 0, 1)
        #self.layout.addWidget(self.btn_open, 1, 3)
        for i in range(4):
            laserGroup = QGroupBox("{} Polarization".format(inv_state_map[i]))
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
                settingsSpinBox.valueChanged.connect(self.settingsChanged)
                aliceSettingsLayout.addWidget(settingsLabel, 0, j)
                aliceSettingsLayout.addWidget(settingsSpinBox, 1, j)
            aliceSettingsGroup.setLayout(aliceSettingsLayout)

            laserLayout.addWidget(self.canvases[i], 0, 0, 1, 1)
            laserLayout.addWidget(self.canvases[i + 4], 1, 0, 1, 1)
            laserLayout.addWidget(aliceSettingsGroup, 2, 0)
            laserGroup.setLayout(laserLayout)

            self.layout.addWidget(laserGroup, 1, i, 1, 1)

        statisticsGroup = QGroupBox("Stats")
        statisticsLayout = QGridLayout()
        settings = [["Frame", (0, 255), 0], ["QBER", (0, 255), 0],
                    ["Sifted Key Rate", (0, 1023), 0],
                    ["Sync Offset", (0, 1023), 0], ["Key match", (0, 1023), 0]]
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
            statisticsLayout.addWidget(statisticsLabel, 0, 2 * j)
            statisticsLayout.addWidget(statisticsText, 0, 1 + 2 * j)
        statisticsGroup.setLayout(statisticsLayout)

        self.layout.addWidget(statisticsGroup, 3, 0, 1, 4)

        # for i, canvas in enumerate(self.canvases):
        #     self.layout.addWidget(canvas, 1 + i, 0)
        self.setLayout(self.layout)

        geometry = app.desktop().availableGeometry()

        self.setGeometry(geometry)  # Thread:

        self.initWorkers()
        # Start Button action:
        self.btn_start.clicked.connect(self.start_threads)

        # Stop Button action:
        self.btn_stop.clicked.connect(self.pause_threads)
        self.btn_open.clicked.connect(self.getfile)

        self.show()

    def updateStats(self, data):
        for text, data in zip(self.statTexts, data):
            text.setText(data)

    def settingsChanged(self, data):
        print("settings changed")
        if self.settingsTimer.isActive():
            self.settingsTimer.stop()
        self.settingsTimer.start(500)

    def updateSettings(self):
        aliceSettings = []
        for p in self.aliceSettingsSpinBoxes:
            aliceSettings.append([])
            for i, sbox in enumerate(p):
                if i == 4:
                    sbox.setRange(0, 1023 - p[i - 1].value())
                aliceSettings[-1].append(sbox.value())
        print("updating settings")
        self.settings_changed_signal.emit(aliceSettings)

    def initWorkers(self):
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())
        evaluator = Evaluator(self.folder, self.key_file)
        self.threadpool.start(evaluator)
        self.stop_signal.connect(evaluator.kill)
        self.pause_signal.connect(evaluator.pause)
        self.start_signal.connect(evaluator.resume)
        evaluator.signals.new_stat_data.connect(self.updateStats,
                                                Qt.QueuedConnection)
        for i in range(4):
            plotter = PhasePlotter(self.canvases[i], i)
            evaluator.signals.new_phase_data.connect(plotter.plot_new_data)
            self.threadpool.start(plotter)
            self.stop_signal.connect(plotter.kill)
            self.pause_signal.connect(plotter.pause)
            self.start_signal.connect(plotter.resume)
        for i in range(4):
            plotter = LinePlotter(self.canvases[i + 4], i)
            evaluator.signals.new_mu_data.connect(plotter.plot_new_data)
            self.threadpool.start(plotter)
            self.stop_signal.connect(plotter.kill)
            self.pause_signal.connect(plotter.pause)
            self.start_signal.connect(plotter.resume)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    key_file = ""
    folder = ""
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
        else:
            filename, file_extension = os.path.splitext(os.path.basename(file))
            if file_extension == ".txt":
                folder = "./data/measure_qber_" + time.strftime(
                    "%Y%m%d-%H%M%S") + "/"
                key_file = folder + filename + file_extension
                os.makedirs(folder)
                shutil.copyfile(file, key_file)

    print(folder, key_file)

    gui = Gui(folder, key_file)
    a = app.exec_()
    gui.stop_threads()
    sys.exit(a)
