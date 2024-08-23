import sys
import random
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QGridLayout,
                             QFileDialog)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QThreadPool, QRunnable, pyqtSlot, QMutex
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

aliceSettings = {
    1: {
        "H": [3, 255, 186, 100, 100 + 76],
        "V": [3, 204, 197, 96, 96 + 70],
        "P": [4, 237, 172, 117, 117 + 66],
        "M": [3, 180, 176, 82, 82 + 63]
    },
    2: {
        "H": [3, 236, 211, 100, 100 + 54],
        "V": [3, 240, 235, 96, 96 + 54],
        "P": [4, 193, 173, 117, 117 + 55],
        "M": [3, 196, 192, 82, 82 + 53]
    },
}

symbol_map = {k: v["ch"] for k, v in channels.items() if k != "CLK"}
inv_symbol_map = {v: k for k, v in symbol_map.items()}

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


def plot_phase(phases,
               filters,
               qber,
               keyrate,
               meas_time,
               t0,
               label,
               frame=None,
               fig=None,
               ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=400)
    else:
        fig.clear()
        gc.collect()
        ax = fig.add_subplot()
    nmax = []
    start = time.time()
    hists = []
    for i in range(0, 4):
        hist, bins = np.histogram(phases[1, phases[0] == i + 1],
                                  bins=100,
                                  range=[0, dt])
        hists.append(hist)
        nmax.append(np.max(hist))
    print(bins)
    print("Histgen took {:.2f}".format(time.time() - start))
    start = time.time()

    for i in range(0, 4):
        ax.bar(bins[:-1], hists[i], width=100, alpha=0.4, label=label[i])
    print("Histplot took {:.2f}".format(time.time() - start))
    start = time.time()
    phase = phases[1, phases[0] == channels["SYNC"]["ch"]]
    n, _ = np.histogram(phase, bins=100, range=(0, dt))
    weight = max(nmax) / max(n)
    _, _, _ = ax.hist(phase,
                      bins=100,
                      range=(0, dt),
                      alpha=0.4,
                      label="SYNC*",
                      weights=np.ones(len(phase)) * weight)
    print("sync Hist took {:.2f}".format(time.time() - start))
    start = time.time()
    ax.vlines(filters[0], 0, max(nmax))
    ax.vlines(filters[1], 0, max(nmax))
    ax.annotate("Δ  = {:.0f}ps".format((filters[1] - filters[0])),
                (filters[0] - 100, max(nmax) * 0.9),
                fontsize="5",
                ha="right",
                va="top")
    ax.annotate("FWHM = {:.0f}ps".format((filters[3] - filters[2])),
                (filters[3] + 100, 20),
                fontsize="5")
    ax.legend()

    ax.set_title(
        "Pulses with mean QBER={:.2f}%\nSifted key rate={:.2f}Kb/s".format(
            qber * 100, keyrate / meas_time / 1000))
    ax.set_xlabel("Detection time in ps")
    ax.set_ylabel("Number of detections")
    fig.tight_layout()
    if frame is None:
        ax.annotate("t_0: {:.1f}s, t_max: {:.1f}s".format(t0, t0 + meas_time),
                    (0, 5),
                    xycoords="figure pixels",
                    fontsize="5")
    else:
        ax.annotate("Frame: {}, t_0: {:.1f}s, t_max: {:.1f}s".format(
            frame, t0, t0 + meas_time), (0, 5),
                    xycoords="figure pixels",
                    fontsize="5")
    print("rest took {:.2f}".format(time.time() - start))
    start = time.time()

    return fig, ax


def guess_key(bins, key_length):
    key = []
    qbers = []
    detections = []
    num_sifted_det = 0
    for i in range(key_length):
        count = bins[i]
        diff1 = count[0] - count[1]
        diff2 = count[2] - count[3]
        diff3 = abs(diff1) - abs(diff2)
        val = 0
        qber = 0
        if diff3 > 0:
            if diff1 > 0:
                val = 0
                qber = count[1] / (count[0] + count[1])
                detection = count[0]
            else:
                val = 1
                qber = count[0] / (count[0] + count[1])
                detection = count[1]
            num_sifted_det += count[0] + count[1]
        else:
            if diff2 > 0:
                val = 2
                qber = count[3] / (count[2] + count[3])
                detection = count[2]
            else:
                val = 3
                qber = count[2] / (count[2] + count[3])
                detection = count[3]
            num_sifted_det += count[2] + count[3]
        key.append(val)
        qbers.append(qber)
        detections.append(detection)

    return np.array(key), np.array(qbers), np.array(detections), num_sifted_det


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


def get_mu(bins, key):
    mus = []
    for i in range(4):
        pol_bin = bins[key == i, :]
        mu = pol_bin.sum() / meas_time / bob_eff / filter / rep_rate / (
            len(pol_bin) / key_length) * alice_trans
        mus.append(mu)
        print("Mean Photon Numbers for {}: {:.3f}".format(
            inv_symbol_map[i + 1], mu))
    return mus


def evaluate_tags(data, channels, offsets, filters=None, verbose=True):
    sync_data = data[1][data[0] == channels["SYNC"]["ch"]]
    sync_phase = int(np.rint(stats.circmean(sync_data % dt, high=dt)))

    key_length = int(np.rint(np.mean(np.diff(sync_data)) / dt))
    if verbose:
        print("Key length: {}".format(key_length))

    if sync_data.size != 0:
        if verbose:
            print("Sync Chan detected")
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
        if verbose:
            print("Got {:.2e} Events in {:.0f}s after first SYNC".format(
                len(data[0]), meas_time))
    else:
        t0 = data[1][1]
        meas_time = (data[1][-1] - t0) * 1E-12
        if verbose:
            print("Got {:.2e} Events in {:.0f}s".format(
                len(data[0]), meas_time))

    if filters is None:
        temp_data = data.copy()
        for i in range(0, 5):
            temp_data[1, temp_data[0] == i + 1] += offsets[i]
        shifts, filters, = find_filter(temp_data % dt)
        offsets += shifts + sync_phase

    for i in range(0, 5):
        data[1, data[0] == i + 1] += offsets[i] - sync_phase

    del sync_phase, sync_data
    phases = data % dt
    time_mask = np.logical_or(
        np.logical_and(phases[1] >= filters[0], phases[1] <= filters[1]),
        data[0] == channels["SYNC"]["ch"])
    data = data[:, time_mask]

    sifted_events = data.copy()[:, data[0] != 5]
    sifted_events[1] = sifted_events[1] // dt

    bins = np.zeros((key_length, 4))
    for i in range(0, 4):
        pos_in_key = ((data[1, data[0] == i + 1] % (dt * key_length)) /
                      dt).astype(int)
        unique_indices, counts = np.unique(pos_in_key, return_counts=True)
        for id, cnt in zip(unique_indices, counts):
            bins[id][i] += cnt

    keys_sent = meas_time / (key_length * dt * 1E-12)

    key_guess, qbers, detections, num_sifted_det = guess_key(bins, key_length)

    det_probs_pol = []
    for i in range(4):
        probs = detections[key_guess == i] / keys_sent
        det_probs_pol.append([probs.mean(), probs.std()])
    det_probs_pol = np.array(det_probs_pol)

    if verbose:
        for i in range(4):
            print(
                "Detection Prob for {}: {:.6f}% with a std of {:.8f} rel: {:.2f}"
                .format(inv_symbol_map[i + 1], det_probs_pol[i][0] * 100,
                        det_probs_pol[i][1] * 100,
                        det_probs_pol[i][0] / np.max(det_probs_pol[:, 0])))

        print("Mean Qber: {:.2f}% with std of {:.4f}".format(
            np.nanmean(qbers) * 100,
            np.nanstd(qbers) * 100))
        print("Max Qber: {:.2f}% at {}".format(
            np.max(qbers) * 100, bins[np.argmax(qbers)]))
        print("Sifted key Rate: {:.2f}".format(num_sifted_det / meas_time))

    mus = []
    for i in range(4):
        pol_bin = bins[key_guess == i, :]
        mu = pol_bin.sum() / meas_time / bob_eff / rep_rate / (
            len(pol_bin) / key_length) * alice_trans
        mus.append(mu)
        if verbose:
            print("Mean Photon Numbers for {}: {:.3f}".format(
                inv_symbol_map[i + 1], mu))

    labels = [
        '{} μ={:.2f}'.format(inv_symbol_map[pol], mu)
        for pol, mu in zip(inv_symbol_map, mus)
    ]

    return [
        offsets, filters,
        [
            phases, filters,
            np.nanmean(qbers), num_sifted_det, meas_time, t0 * 1E-12, labels
        ], sifted_events
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


def check_key(key_filename, key):
    target_key = []
    with open(key_filename, "r") as file:
        for char in file.readline()[:-1]:
            target_key.append(symbol_map[char.upper()] - 1)

    temp_key = target_key + target_key
    sames = []
    for i in range(len(target_key)):
        comp_key = temp_key[i:i + len(target_key)]
        same = 0
        for j in range(len(target_key)):
            if (key[j] == comp_key[j]):
                same += 1
        sames.append(same)
    if (max(sames) == len(target_key)):
        ind_offset = np.argmax(sames)
        print("Keys match with offset: {}".format(ind_offset))
    #else:
    #    print(len(sames), max(sames), np.argmax(sames))
    #    shift_key = temp_key[np.argmax(sames):np.argmax(sames) +
    #                         len(target_key)]
    #    sumcounts = [[], [], [], []]
    #    for i in range(len(target_key)):
    #        if (key[i] != shift_key[i]):
    #            print(i, key[i], shift_key[i], bins[i], np.sum(bins[i]))
    #        sumcounts[key[i]].append(np.sum(bins[i]))
    #    for i in range(4):
    #        print(i, np.mean(sumcounts[i]))


def main():
    file = sys.argv[1]
    if not os.path.exists(file):
        print("Cannot open {}".format(file))
        sys.exit()
    folder = os.path.dirname(file) + "/"
    filename, file_extension = os.path.splitext(os.path.basename(file))

    if file_extension == ".txt":
        folder = "./data/measure_qber_" + time.strftime("%Y%m%d-%H%M%S") + "/"
        filename = folder + filename
        key_filename = filename + file_extension
        os.makedirs(folder)
        shutil.copyfile(file, key_filename)
        # Turn on Alice:
        source = AliceLmu(host="t14s", aliceSettings=aliceSettings)
        source.turn_on()
        source.send_key(key_filename)
        # Measure clicks using Timetagger:
        timestamp = TimeTaggerUltra(channels)
        ts = timestamp.read(120)
        save_ts(ts, filename + ".bin")
        source.stop_key()
    elif file_extension == ".ttbin":
        # Measure clicks using Virtual Timetagger:
        timestamp = TimeTaggerUltraVirtual(channels, file)
        ts = timestamp.read(10)
        save_ts(ts, filename + ".bin")
        key_filename = folder + filename + ".txt"
    else:
        ts = load_ts(file)
        key_filename = folder + filename + ".txt"

    rd = ts.view(np.int64)

    # simulate channel loss
    np.random.seed(42)
    nr = 0
    e_id = 0
    for i in range(1, nr):
        b_id = int(np.random.random() * (len(rd[0]) / nr)) + e_id
        e_id = int(np.random.random() * (len(rd[0]) / nr) + b_id)
        shift = int(np.random.random() * 40000 + 500)
        print(b_id, e_id, shift)
        rdt = rd[:, b_id:e_id]
        rdt = rdt[:, rdt[0] != 5]
        rde = rd[:, e_id:]
        rde[1] += shift
        rd = np.concatenate([rd[:, :b_id], rdt, rde], axis=1)
        print(len(rd[0]))
    print((np.diff(rd[1]) > 0).all())

    # simulate live data
    rd[1] -= rd[1][0]
    end = int(rd[1][-1] * 1E-12)
    #datas = [
    #    rd[:, np.logical_and(rd[1] > (i) * 1E12, rd[1] < (i + 1) * 1E12)]
    #    for i in range(2)
    #]
    #for i, data in enumerate(datas):
    #    save_ts(data, "frame{}.bin".format(i))
    datas = [rd]
    offsets = chan_offset
    fig, ax = plt.subplots(figsize=(5, 3), dpi=400)
    plt.show(block=False)
    plt.pause(0.1)

    try:
        frame = -1
        while True:
            frame += 1
            start = time.time()
            data = datas[frame % len(datas)]
            if len(data[0]) <= 0:
                print("No Timestamps in frame")
                continue
            data = get_valid_frames(data, verbose=False)
            start2 = time.time()
            print("Valid Frames took: {:.2f}".format(start2 - start))
            if len(data) <= 0:
                print("No Sync detected in frame")
                continue
            offsets, filters, plot_data, sifted_events = evaluate_tags(
                data, channels, offsets, verbose=False)
            start3 = time.time()
            print("Evaluate took: {:.2f}".format(start3 - start2))

            plot_phase(*plot_data, frame=frame, fig=fig, ax=ax)
            fig.canvas.flush_events()
            start4 = time.time()
            print("Plot took: {:.2f}".format(start4 - start3))
            plt.pause(.1)
            rgba = np.asarray(fig.canvas.buffer_rgba())
            im = Image.fromarray(rgba)
            im.save(folder + "phases{}.png".format(frame))
            start5 = time.time()
            print("Image took: {:.2f}".format(start5 - start4))
            save_sifted(sifted_events, folder + "frame{}.csv".format(frame))
            start6 = time.time()
            print("Sifted took: {:.2f}".format(start6 - start5))
            duration = time.time() - start
            print("Frame {} done. {:.1f}fps".format(frame, 1 / duration))
            duration = time.time() - start
            del plot_data
            if frame >= len(datas):
                while plt.get_fignums():
                    plt.pause(0.2)
                break
    except KeyboardInterrupt:
        print("Exiting")


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
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
            self.signals.progress.emit(i)
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


class Plotter(Worker):

    def __init__(self, canvas, nr):
        Worker.__init__(self)
        self.nr = nr
        self.canvas = canvas
        self.plot_data = []
        self.mutex = QMutex()
        self.has_new_data = False

    def loop(self, i):
        self.mutex.lock()
        if self.has_new_data:
            frame = i
            hists, bins, filters, label = self.plot_data
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot()
            start = time.time()
            hist = hists[self.nr]
            hist_other = np.sum(
                [hist for i, hist in enumerate(hists[:-1]) if i != self.nr],
                axis=0) / 3
            ax.bar(bins[:-1],
                   hist,
                   width=100,
                   alpha=0.4,
                   label=label[self.nr],
                   color="C{}".format(self.nr))
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
            #self.canvas.fig.tight_layout()
            print("plot took {:.2f}s".format(time.time() - start))
            start = time.time()
            self.canvas.draw()
            self.has_new_data = False
        self.mutex.unlock()

    def plot_new_data(self, data):
        self.mutex.lock()
        self.plot_data = data
        self.has_new_data = True
        self.mutex.unlock()


class Evaluator(Worker):
    new_phase_data = pyqtSignal(object)

    def __init__(self):
        Worker.__init__(self)
        self.mutex = QMutex()
        self.maxi = 118
        self.offsets = chan_offset

    def loop(self, i):
        start = time.time()
        data = load_ts("data/gui_test/frame{}.bin".format(i % self.maxi))
        data = data.view(np.int64)

        if len(data[0]) <= 0:
            print("No Timestamps in frame")
            return
        data = get_valid_frames(data, verbose=False)
        if len(data) <= 0:
            print("No Sync detected in frame")
            return
        self.offsets, filters, plot_data, sifted_events = evaluate_tags(
            data, channels, self.offsets, verbose=False)
        phases = plot_data[0]
        nmax = []
        hists = []
        for i in range(0, 5):
            hist, bins = np.histogram(phases[1, phases[0] == i + 1],
                                      bins=100,
                                      range=[0, dt])
            hists.append(hist)
            nmax.append(np.max(hist))
        hists[-1] = hists[-1] * (np.mean(nmax[:-1]) / nmax[-1])
        self.signals.new_phase_data.emit(
            [hists, bins, plot_data[1], plot_data[-1]])
        duration = time.time() - start

        print("eval took {:.2f}s".format(duration))
        time.sleep(max(0, 1 - duration))


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class Gui(QWidget):
    stop_signal = pyqtSignal()
    pause_signal = pyqtSignal()
    start_signal = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(Gui, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.threads = {}

        #self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        #self.timer = QtCore.QTimer()
        #self.timer.setInterval(100)
        #self.timer.timeout.connect(self.update_plot)
        #self.timer.start()
        self.initUI()

    def initUI(self):
        self.canvases = []
        for i in range(4):
            self.canvases.append(MplCanvas(self, width=5, height=4, dpi=100))

        # Buttons:
        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_open = QPushButton('Open')

        # GUI title, size, etc...
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('ThreadTest')
        self.layout = QGridLayout()
        self.layout.addWidget(self.btn_start, 0, 0)
        self.layout.addWidget(self.btn_stop, 0, 1)
        self.layout.addWidget(self.btn_open, 1, 3)
        for i, canvas in enumerate(self.canvases):
            self.layout.addWidget(canvas, 1 + i, 0)
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

    def initWorkers(self):
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())
        evaluator = Evaluator()
        self.threadpool.start(evaluator)
        self.stop_signal.connect(evaluator.kill)
        self.pause_signal.connect(evaluator.pause)
        self.start_signal.connect(evaluator.resume)
        for i in range(4):
            plotter = Plotter(self.canvases[i], i)
            evaluator.signals.new_phase_data.connect(plotter.plot_new_data)
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
    gui = Gui()
    a = app.exec_()
    gui.stop_threads()
    sys.exit(a)
