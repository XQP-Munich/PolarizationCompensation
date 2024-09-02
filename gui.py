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
import concurrent.futures

from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QGridLayout,
    QFileDialog,
    QLabel,
    QGroupBox,
    QSpinBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTextEdit,
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

from Devices.TimeTaggerUltra import TimeTaggerUltra
from Devices.AliceLmu import AliceLmu
from led_widget import LedIndicator

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
    [3, 230, 233, 700, 54],
    [3, 231, 235, 696, 54],
    [4, 190, 186, 717, 55],
    [4, 163, 176, 682, 59],
]
# 70 / 30
aliceSettings = [
    [3, 230, 238, 700, 54],
    [3, 236, 241, 696, 54],
    [4, 189, 184, 717, 55],
    [4, 159, 173, 682, 58],
]
# new
aliceSettings = [
    [3, 245, 238, 700, 54],
    [3, 236, 241, 696, 54],
    [4, 188, 184, 717, 55],
    [4, 171, 173, 682, 53],
]
symbol_map = {k: v["ch"] for k, v in channels.items() if k != "CLK"}
inv_symbol_map = {v: k for k, v in symbol_map.items()}

state_map = {"H": 0, "V": 1, "P": 2, "M": 3, "h": 4, "v": 5, "p": 6, "m": 7}
inv_state_map = {v: k for k, v in state_map.items()}

filter = 1.32 / 100
bob_eff = filter * 25.3852 / 100
alice_trans = 0.917 / 0.9954
rep_rate = 100E6
chan_offset = [2000, 2071, -4515, 3070, 2000]
dt = 10000
meas_time = 0
key_length = 0


def save_ts(ts, path):
    data = np.bitwise_or(ts[0], np.left_shift(ts[1], 4))
    data.tofile(path)


import random


def save_sifted(sifted, path):
    print("saving {} events".format(len(sifted[0])))
    pol = []
    for chan in sifted[0]:
        if chan == 1:
            pol.append("H")
            #pol.append("P")
        if chan == 2:
            pol.append("V")
            #pol.append("M")
        if chan == 3:
            pol.append("P")
            #pol.append("H")
        if chan == 4:
            pol.append("M")
            #pol.append("V")
    with open(path, "w") as f:
        for i in range(len(sifted[1])):
            #if pol[i] == "P" or pol[i] == "M":
            #    if random.random() > 0.42857:
            #        continue
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


def evaluate_tags(data,
                  channels,
                  offsets,
                  sent_key=None,
                  key_length=None,
                  sync_offset=np.nan,
                  filters=None,
                  verbose=True,
                  time_filtering=True):
    sync_data = data[1][data[0] == channels["SYNC"]["ch"]]
    sync_phase = 0
    #sync_offset = 6
    if sent_key is not None:
        key_length = len(sent_key)
    if sync_data.size != 0:
        key_length_sync = int(np.rint(np.mean(np.diff(sync_data)) / dt))
        if key_length is not None and key_length_sync != key_length:
            print("Target key length {} does not match detected key length {}".
                  format(key_length, key_length_sync))
            return []
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
        data[1] -= t0
        data = data[:, data[1] >= 0]
        sync_phase = 0
        if key_length is None:
            key_length = 1

    if verbose:
        print("Got {:.2e} Events in {:.0f}s".format(len(data[0]), meas_time))

    if filters is None:
        temp_data = data.copy()
        for i in range(0, 5):
            temp_data[1, temp_data[0] == i + 1] += offsets[i]
        try:
            shifts, filters, = find_filter(temp_data % dt)
            offsets += shifts + sync_phase
        except:
            print("Unable to find peaks")
            offsets = np.array(offsets) + sync_phase
            filters = np.array([0, dt, 0, dt])

    if not time_filtering:
        filters = np.array([0, dt, 0, dt])

    if verbose:
        print("Offsets: {}".format(offsets))

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
    keymatch = None
    if sent_key is not None and np.isnan(sync_offset):
        matching_symbols, sync_offset = compare_key(sent_key, key_guess)
        if verbose:
            if (matching_symbols == key_length):
                print("Keys match with offset: {}".format(sync_offset))
            else:
                keymatch = "{}/{}".format(matching_symbols, key_length)

                print("Keys match with {}/{} symbols, most likely offset: {}".
                      format(matching_symbols, key_length, sync_offset))
                if matching_symbols <= key_length * 0.50:
                    sync_offset = np.nan

    used_key = key_guess
    if sent_key is not None and not np.isnan(sync_offset):
        print("using sent_key")
        used_key = sent_key
        used_key = np.array(
            (used_key + used_key)[sync_offset:sync_offset + key_length])

    mcount = [[], [], [], [], [], [], [], []]
    if False:
        for i, b in enumerate(bins):
            if used_key[i] != key_guess[i]:
                print("*", end="")
            if b.sum() <= np.mean(np.sum(bins, axis=1)) / 2:
                print("#", end="")
            if b.sum() >= np.mean(np.sum(bins, axis=1)) * 2:
                print("%", end="")
            print(inv_state_map[used_key[i]], inv_state_map[key_guess[i]], b,
                  "{:.2f}".format(qbers[i] * 100), num_sifted_det[i])
            mcount[used_key[i]].append(b.sum())
        for c in mcount:
            print(np.mean(c), np.std(c))
        for c in mcount:
            print(c)
    if sent_key is not None and not np.isnan(sync_offset):
        phases[2] = used_key[phases[2]]
    else:
        phases[2] = phases[0]

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
        mu = pol_bin.sum(axis=1) / meas_time / bob_eff / rep_rate / (
            1 / key_length) * alice_trans
        mus.append(mu)

    return [
        phases, offsets, filters, sync_offset, qbers, num_sifted_det,
        meas_time, t0 * 1E-12, sifted_events, mus, keymatch
    ]


def process_data(data,
                 frame,
                 offsets=chan_offset,
                 sync_offset=np.nan,
                 sent_key=None,
                 verbose=True,
                 time_filtering=True):
    print("Evaluating Frame {}".format(frame))
    start = time.time()

    if len(data[0]) <= 0:
        print("No Timestamps in frame")
        return frame, None
    valid_data = get_valid_frames(data, verbose=False)
    if len(valid_data) <= 0:
        print("No Sync detected in frame")
        if np.isnan(sync_offset):
            valid_data = data
        else:
            print("Skipping Frame")
            return frame, None
    et = evaluate_tags(valid_data,
                       channels,
                       offsets,
                       sync_offset=sync_offset,
                       sent_key=sent_key,
                       verbose=verbose,
                       time_filtering=time_filtering)
    if len(et) == 0:
        print("Evaluation error, continue")
        return frame, None
    phases, offsets, filters, sync_offset = et[:4]
    qbers, num_sifted_det, meas_time, t0 = et[4:8]
    sifted_events, mus, key_match = et[8:]

    #self.verbose = False
    nmax = []
    hists = []
    bins = []
    phases_pol = phases[:, phases[0] != 5]
    for pol in range(8):
        hist, bins = np.histogram(phases_pol[1, phases_pol[2] == pol],
                                  bins=100,
                                  range=(0, dt))
        hists.append(hist / 1000)
        nmax.append(np.max(hist) / 1000)
    hist, bins = np.histogram(phases[1, phases[0] == 5],
                              bins=100,
                              range=(0, dt))
    hists.append(hist)
    nmax.append(np.max(hist))

    hists[-1] = hists[-1] * (np.max(nmax[:-1]) / nmax[-1])
    phase_data = [hists, bins, filters]
    mu_data = [frame, mus]
    ui_data = [
        "{}".format(frame),
        "{:.2f}% ± {:.2f}%".format(np.mean(qbers) * 100,
                                   np.std(qbers) * 100),
        "{:.2f}Kbit/s".format(np.sum(num_sifted_det) / meas_time / 1000),
        "{:.0f}".format(sync_offset),
        "{}".format(offsets),
        key_match,
    ]
    eval_data = [offsets, sync_offset]
    duration = time.time() - start

    print("eval took {:.2f}s".format(duration))
    return frame, [eval_data, phase_data, mu_data, ui_data, sifted_events]


def get_valid_frames(data, verbose=True):
    valid_timing = 500  # in ps
    sync_indices = np.where(data[0] == channels["SYNC"]["ch"])[0]
    if len(sync_indices) <= 20:
        return []
    sync_diffs = np.diff(data[1][sync_indices], append=0)
    sync_diffs_median = np.median(sync_diffs)
    sync_dropouts = np.where(
        np.logical_or(sync_diffs > (sync_diffs_median + valid_timing),
                      sync_diffs < (sync_diffs_median - valid_timing)))[0]
    if len(sync_dropouts) > 1 and verbose:
        print("{} SYNC dropouts detected:\n{}".format(
            len(sync_dropouts), sync_indices[sync_dropouts]))
    sync_dropouts = np.concatenate(([-1], sync_dropouts))
    valid_frames = []
    for i in range(len(sync_dropouts) - 1):
        beg = sync_indices[sync_dropouts[i] + 1]
        end = sync_indices[sync_dropouts[i + 1]]
        if end > beg:
            d = data[:, beg:end + 1]
            valid_frames.append(d)
    return np.concatenate(valid_frames, axis=1)


def compare_key(sent_key, detected_key):
    sames = []
    for i in range(len(sent_key)):
        sames.append(
            np.count_nonzero(detected_key == np.roll(sent_key, -i) % 4))
    return max(sames), np.argmax(sames)


def plot_phases(canvas, plot_data):
    hists, bins, filters = plot_data
    ax = canvas.axes
    ax.cla()
    start = time.time()
    hist_sig = hists[canvas.pol]
    hist_dec = hists[canvas.pol + 4]
    hist_other = np.sum([
        hist for i, hist in enumerate(hists[:-1])
        if (i != canvas.pol and i != canvas.pol + 4)
    ],
                        axis=0) / 3
    ax.bar(bins[:-1],
           hist_sig,
           width=100,
           alpha=0.4,
           label=inv_symbol_map[canvas.pol + 1] + " Signal",
           color="C{}".format(canvas.pol))
    ax.bar(bins[:-1],
           hist_dec,
           width=100,
           alpha=0.4,
           label=inv_symbol_map[canvas.pol + 1] + " Decoy",
           color="C{}".format(canvas.pol + 4))
    ax.bar(bins[:-1],
           hist_other,
           width=100,
           alpha=0.2,
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
    canvas.fig.tight_layout()
    print("phase plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


def plot_mus(canvas, plot_data):
    frame, mus, = plot_data
    ax = canvas.axes
    ax.cla()
    start = time.time()
    if canvas.xdata is None:
        canvas.xdata = []
        canvas.ydata = []
    if len(canvas.xdata) >= 10:
        canvas.xdata = canvas.xdata[1:]
        canvas.ydata = canvas.ydata[1:]
    canvas.xdata.append(frame)
    canvas.ydata.append([
        np.mean(mus[canvas.pol]),
        np.mean(
            [np.mean(mu) for i, mu in enumerate(mus[:4]) if i != canvas.pol]),
        np.mean(mus[canvas.pol + 4]),
        np.mean(
            [np.mean(mu) for i, mu in enumerate(mus[4:]) if i != canvas.pol])
    ])
    ax.plot(canvas.xdata, canvas.ydata, label=["μ", "μ_r", "ν", "ν_r"])
    ax.legend(loc="center left")
    ax.hlines([0.51, 0.15],
              canvas.xdata[0],
              canvas.xdata[-1],
              linestyles="dashed",
              colors="gray")

    ax.set_title("μ={:.2f}±{:.2f}  ν={:.2f}±{:.2f}".format(
        np.mean(mus[canvas.pol]), np.std(mus[canvas.pol]),
        np.mean(mus[canvas.pol + 4]), np.std(mus[canvas.pol + 4])))

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean photon number")
    canvas.fig.tight_layout()
    print("mu plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


class WorkerSignals(QObject):
    new_stat_data = pyqtSignal(object)


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

    def loop(self, i):
        pass

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def kill(self):
        self.is_running = False


class Experimentor(Worker):

    def __init__(self, canvases, mode, folder, key_file=""):
        Worker.__init__(self)
        # Experimentor vars
        self.mode = mode
        self.folder = folder
        self.key_file = key_file
        self.canvases = canvases
        self.sent_key = None
        if self.key_file != "":
            self.get_key()

        self.save_raw = False
        self.save_sifted = False
        self.time_filtering = True

        self.host = "t14s"

        # Evaluator vars
        self.sync_offset = np.nan
        self.offsets = chan_offset

        self.reset = False

        self.meas_time = 5

        self.frame = 0
        self.last_plotted_frame = -1
        self.eval_lock = QReadWriteLock()
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.futures = []

        if self.mode == 0:
            self.initPlayback()
        else:
            self.initMeasurement()

    def get_key(self):
        self.sent_key = []

        with open(self.key_file, "r") as file:
            for char in file.readline()[:-1]:
                self.sent_key.append(state_map[char])

    def initPlayback(self):
        self.data_files = []
        self.get_data_files()
        self.time_last = time.time()
        self.loop_playback = True

    def initMeasurement(self):
        self.alice = AliceLmu(self.host)
        self.timestamp = TimeTaggerUltra(channels)
        self.stream = self.timestamp.get_stream()
        self.stream.start()

    def set_aliceSettings(self, settings):
        aliceSettings = {0: {}}
        pols = ["H", "V", "P", "M"]
        for i, pol in enumerate(settings):
            pol = pol[:-1] + [pol[-2] + pol[-1]]
            aliceSettings[0][pols[i]] = pol
        print(aliceSettings)
        self.alice = AliceLmu(self.host, aliceSettings=aliceSettings)
        self.alice.turn_off()
        self.alice.turn_on(set=0)
        self.reset = True
        self.reset_eval_settings()

    def send_key(self):
        self.alice.send_key(self.key_file)
        self.reset = True
        self.reset_eval_settings()

    def reset_eval_settings(self):
        self.offsets = chan_offset
        self.sync_offset = np.nan
        self.verbose = True

    def update_eval_settings(self, settings):
        if settings is not None:
            self.time_filtering, self.save_raw, self.save_sifted = settings

    def get_data_files(self):
        for f in os.listdir(self.folder):
            if f.endswith(".bin"):
                self.data_files.append(f)
        print(self.data_files)

    def loop(self, i):
        data = None
        if self.mode == 0:
            if self.reset or time.time() - self.time_last >= self.meas_time:
                if self.loop_playback:
                    frame = self.frame % len(self.data_files)
                else:
                    if self.frame >= len(self.data_files):
                        print("No more Frames to read")
                        self.kill()
                        return
                    frame = self.frame
                print("Reading Frame {}".format(frame))
                data = load_ts(self.folder + self.data_files[frame]).view(
                    np.int64)
                self.time_last = time.time()
        else:
            if self.reset or self.stream.getCaptureDuration(
            ) * 1E-12 >= self.meas_time:
                print("Measured Frame {}".format(self.frame))
                data = self.stream.getData()
                data = [data.getChannels(), data.getTimestamps()]
                clock_errors = self.timestamp.get_clock_errors()
                if clock_errors != 0:
                    print("{} Clock errors detected".format(clock_errors))

        if data is not None and not self.reset:
            future = self.executor.submit(process_data,
                                          data,
                                          self.frame,
                                          offsets=self.offsets,
                                          sync_offset=self.sync_offset,
                                          sent_key=self.sent_key,
                                          time_filtering=self.time_filtering,
                                          verbose=True)
            future.add_done_callback(self.eval_done)
            self.futures.append(future)
            if self.save_raw:
                future = self.executor.submit(
                    save_ts, data,
                    self.folder + "raw_frame{}.bin".format(self.frame))
                self.futures.append(future)
            self.frame += 1

        for future in list(self.futures):
            if future.cancelled():
                print("got cancelled")
            if future.done():
                future.result()
                self.futures.remove(future)
        self.reset = False

    def eval_done(self, future):
        print("Eval done")
        frame, res = future.result()

        lpf = -1
        while lpf + 1 != frame:
            time.sleep(0.1)
            self.eval_lock.lockForRead()
            lpf = self.last_plotted_frame
            self.eval_lock.unlock()

        if res is not None:
            eval_data, phase_data, mu_data, ui_data, sifted_events = res

            for i in range(4):
                future = self.executor.submit(plot_phases, self.canvases[i],
                                              phase_data)
                self.futures.append(future)
            for i in range(4):
                future = self.executor.submit(plot_mus, self.canvases[i + 4],
                                              mu_data)
                self.futures.append(future)
            if self.save_sifted:
                future = self.executor.submit(
                    save_sifted, sifted_events,
                    self.folder + "sifted_frame{}.csv".format(frame))
                self.futures.append(future)

            self.offsets, self.sync_offset = eval_data
            self.signals.new_stat_data.emit(ui_data)
        self.eval_lock.lockForWrite()
        self.last_plotted_frame = frame
        self.eval_lock.unlock()


class MplCanvas(FigureCanvas):

    def __init__(self, pol=0, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.pol = pol
        self.xdata = None
        self.ydata = None
        super(MplCanvas, self).__init__(self.fig)


class Gui(QWidget):
    stop_signal = pyqtSignal()
    pause_signal = pyqtSignal()
    start_signal = pyqtSignal()
    alice_settings_set_signal = pyqtSignal(object)
    eval_settings_changed_signal = pyqtSignal(object)

    def __init__(self, mode, folder, key_file, *args, **kwargs):
        super(Gui, self).__init__(*args, **kwargs)

        self.folder = folder
        self.key_file = key_file
        self.aliceSettingsSpinBoxes = []
        self.settingsTimer = QTimer()
        self.settingsTimer.setSingleShot(True)
        self.settingsTimer.timeout.connect(self.updateAliceSettings)
        self.aliceSettings = aliceSettings
        self.frame = 0
        self.mode = mode
        self.initUI()

    def initUI(self):
        self.canvases = []
        for i in range(8):
            self.canvases.append(
                MplCanvas(pol=i % 4, width=4, height=5, dpi=100))

        # Buttons:        self.btn_open = QPushButton('Open')
        self.btn_set_Alice = QPushButton('Set Alice')
        self.btn_set_Alice.clicked.connect(self.setAliceSettings)
        self.btn_set_Key = QPushButton('Set Key')

        # Alice settings

        # GUI title, size, etc...
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle("Live Analysis")
        self.gridlayout = QGridLayout()

        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_open = QPushButton('Open')

        self.btn_widget = QWidget()
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.btn_start)
        self.btn_layout.addWidget(self.btn_stop)
        self.btn_layout.addWidget(self.btn_open)
        self.btn_widget.setLayout(self.btn_layout)
        self.gridlayout.addWidget(self.btn_widget, 0, 0)

        self.set_widget = QWidget()
        self.set_layout = QHBoxLayout()
        self.set_widget.setLayout(self.set_layout)

        self.host_widget = QWidget()
        self.host_layout = QVBoxLayout()
        self.host_widget.setLayout(self.host_layout)
        self.host_label = QLabel("Alice Host")
        self.host_text = QLineEdit("local")
        self.host_text.setMaximumWidth(10)
        self.host_layout.addWidget(self.host_label)
        self.host_layout.addWidget(self.host_text)

        self.set_layout.addWidget(self.host_widget)

        #self.gridlayout.addWidget(self.set_widget, 0, 1)

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

        self.initWorkers()
        # Start Button action:
        self.btn_start.clicked.connect(self.start_threads)

        # Stop Button action:
        self.btn_stop.clicked.connect(self.pause_threads)
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

    def updateEvaluationSettings(self):
        self.eval_settings_changed_signal.emit([
            self.check_time_filter.isChecked(),
            self.check_save_raw.isChecked(),
            self.check_save_sifted.isChecked()
        ])

    def initWorkers(self):
        self.threadpool = QThreadPool.globalInstance()
        experimentor = Experimentor(self.canvases, self.mode, self.folder,
                                    self.key_file)
        self.threadpool.start(experimentor)
        self.stop_signal.connect(experimentor.kill)
        self.pause_signal.connect(experimentor.pause)
        self.start_signal.connect(experimentor.resume)
        self.btn_set_Key.clicked.connect(experimentor.send_key)
        self.alice_settings_set_signal.connect(experimentor.set_aliceSettings)
        self.eval_settings_changed_signal.connect(
            experimentor.update_eval_settings)
        experimentor.signals.new_stat_data.connect(self.updateStats)

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

    gui = Gui(mode, folder, key_file)
    a = app.exec_()
    gui.stop_threads()
    sys.exit(a)
