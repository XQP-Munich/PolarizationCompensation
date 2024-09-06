import time
import os
import numpy as np
import concurrent.futures
import socket

from PyQt5.QtCore import (
    QReadWriteLock,
    QThread,
    QObject,
    pyqtSignal,
    QRunnable,
    pyqtSlot,
    QThreadPool,
)

from QKD.Evaluation import process_data
from QKD.Plotting import plot_phases, plot_mus

chan_offset = [12000, 12071, 10000 - 4515, 13070, 12000]  # todo remove global

state_map = {"H": 0, "V": 1, "P": 2, "M": 3, "h": 4, "v": 5, "p": 6, "m": 7}
inv_state_map = {v: k for k, v in state_map.items()}


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

    def __init__(self,
                 canvases,
                 mode,
                 folder,
                 channels,
                 alice=None,
                 timestamp=None,
                 key_file=""):
        Worker.__init__(self)
        # Experimentor vars
        self.mode = mode
        self.folder = folder
        self.key_file = key_file
        self.canvases = canvases
        self.channels = channels
        self.alice = alice
        self.timestamp = timestamp
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
        self.lastmeas = 0

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
        print("Reading in Keyfile: {}".format(self.key_file))
        with open(self.key_file, "r") as file:
            for char in file.readline()[:-1]:
                self.sent_key.append(state_map[char])

    def set_meas_time(self, time):
        print("Setting measurement time to {}s".format(time))
        self.meas_time = time

    def initPlayback(self):
        self.data_files = []
        self.get_data_files()
        self.time_last = time.time()
        self.loop_playback = True

    def initMeasurement(self):
        self.stream = self.timestamp.get_stream()
        self.stream.start()

    def set_aliceSettings(self, settings):
        aliceSettings = {0: {}}
        pols = ["H", "V", "P", "M"]
        for i, pol in enumerate(settings):
            pol = pol[:-1] + [pol[-2] + pol[-1]]
            aliceSettings[0][pols[i]] = pol
        print(aliceSettings)
        self.alice.set_Settings(aliceSettings=aliceSettings)
        #self.alice.turn_off()
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
            if self.reset or (self.stream.getCaptureDuration() * 1E-12 -
                              self.lastmeas) >= self.meas_time:
                self.lastmeas = self.stream.getCaptureDuration() * 1E-12
                print("Measured Frame {}".format(self.frame))
                data = self.stream.getData()
                data = np.array([data.getChannels(), data.getTimestamps()])
                clock_errors = self.timestamp.get_clock_errors()
                if clock_errors != 0:
                    print("{} Clock errors detected".format(clock_errors))

        if data is not None and not self.reset:
            future = self.executor.submit(process_data,
                                          data,
                                          self.frame,
                                          self.channels,
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


class Connection(Worker):

    def __init__(self, sock, address, parent=None):
        Worker.__init__(self)
        print("New Connection from: {}".format(address))
        self.sock = sock
        self.address = address
        self.parent = parent

    def loop(self, i):
        print(i)
        try:
            data = self.sock.recv(32)
            if len(data) == 0:
                raise Exception()
        except Exception as e:
            print(e)
            print("Client " + str(self.address) + " has disconnected")
            if self.parent != None:
                self.parent.connected = False
            self.kill()
            return
        if data != "":
            print(str(data.decode("utf-8")))


class Server(Worker):

    def __init__(self, host="localhost", port=31415):
        Worker.__init__(self)
        self.host = host
        self.port = port
        self.threadpool = QThreadPool.globalInstance()
        self.start_server()

    def start_server(self):
        print("Starting server on: {}:{}".format(self.host, self.port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)

    def loop(self, i):
        try:
            sock, address = self.sock.accept()
            conn = Connection(sock, address)
            self.threadpool.start(conn)
        except Exception as e:
            print(e)


class Client(Worker):

    def __init__(self, host="localhost", port=31415):
        Worker.__init__(self)
        self.host = host
        self.port = port
        self.connected = False
        self.threadpool = QThreadPool.globalInstance()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.conn = Connection(self.sock, "client", parent=self)
            self.threadpool.start(self.conn)
            self.connected = True
        except:
            self.connected = False
            print("Could not make a connection to the server")

    def loop(self, i):
        if not self.connected:
            self.connect()
        else:
            print(i)
            if i % 4 == 0:
                self.sock.send("{}".format(i).encode())
