import time
import os
import numpy as np
import concurrent.futures
import socket
from io import BytesIO
import struct
import re

from PyQt5.QtCore import (
    QMutex,
    QReadWriteLock,
    QThread,
    QObject,
    pyqtSignal,
    QRunnable,
    pyqtSlot,
    QThreadPool,
    QTimer,
)
from PyQt5.QtTest import QSignalSpy
from QKD.Evaluation import process_data
from QKD.Plotting import *

chan_offset = [12000, 12071, 10000 - 4515, 13070, 12000]  # todo remove global

state_map = {"H": 0, "V": 1, "P": 2, "M": 3, "h": 4, "v": 5, "p": 6, "m": 7}
inv_state_map = {v: k for k, v in state_map.items()}


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


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


def save_mu(frame, mu, path):
    with open(path, "a") as f:
        f.writelines("{}\t{}\n".format(frame, mu))


def load_mu(path):
    if os.path.isfile(path):
        mus = []
        with open(path, "r") as f:
            for line in f.readlines():
                frame, mu = line[:-1].split("\t")
                mus.append([int(frame), float(mu)])
        return np.array(mus)


class WorkerSignals(QObject):
    new_data = pyqtSignal(object)
    conn_closed = pyqtSignal(object)

    def __init__(self):
        super().__init__()


class Worker(QRunnable):

    def __init__(self):
        super().__init__()

        self.signals = WorkerSignals()
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
        super().__init__()
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

        self.floss = -25.1
        self.fmu = 0.33
        self.last_valid_sync = 0
        self.alice_raw_mu_data = []
        self.alice_mu_data = []
        self.alice_mu_history = None
        self.plot_mode = 0
        if self.mode == 0:
            self.initPlayback()
        else:
            self.initMeasurement()

    def initPlayback(self):
        self.data_files = []
        self.get_data_files()
        self.get_aux_data()
        self.time_last = time.time()
        self.loop_playback = True

    def initMeasurement(self):
        self.stream = self.timestamp.get_stream()
        self.stream.start()

    def get_data_files(self):
        print("Reading in data files from: {}".format(self.folder))
        for f in os.listdir(self.folder):
            if f.endswith(".bin"):
                self.data_files.append(f)
        self.data_files.sort(key=natural_keys)
        print(self.data_files)

    def get_key(self):
        self.sent_key = []
        print("Reading in Keyfile: {}".format(self.key_file))
        with open(self.key_file, "r") as file:
            for char in file.readline()[:-1]:
                self.sent_key.append(state_map[char])

    def get_aux_data(self):
        print("Reading in auxilary files from: {}".format(self.folder))
        self.alice_mu_history = load_mu(self.folder + "alice_Brightness.csv")

    def set_meas_time(self, time):
        print("Setting measurement time to {}s".format(time))
        self.meas_time = time

    def set_aliceSettings(self, settings):
        aliceSettings = {0: {}}
        pols = ["H", "V", "P", "M"]
        for i, pol in enumerate(settings):
            pol = pol[:-1] + [pol[-2] + pol[-1]]
            aliceSettings[0][pols[i]] = pol
        print(aliceSettings)
        self.alice.set_aliceSettings(aliceSettings)
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

    def handle_settings_change(self, settings):
        reprate, alice_loss, bob_loss, plot_mu, plot_floss, plot_lloss, fmu, floss = settings
        if plot_mu:
            self.plot_mode = 0
        elif plot_floss:
            self.plot_mode = 1
        elif plot_lloss:
            self.plot_mode = 2
        self.reprate = reprate
        self.alice_loss = alice_loss
        self.bob_loss = bob_loss
        self.fmu = fmu
        self.floss = floss
        for canv in self.canvases:
            canv.resetData()

    def update_eval_settings(self, settings):
        if settings is not None:
            self.time_filtering, self.save_raw, self.save_sifted = settings

    def handle_alice_counts(self, counts):
        print("New Alice mu data: {}".format(counts))
        self.alice_raw_mu_data.append(counts)

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
                print("Reading Frame {} from {}".format(
                    frame, self.data_files[frame]))
                data = load_ts(self.folder + self.data_files[frame]).view(
                    np.int64)
                if self.alice_mu_history is not None:
                    mus = self.alice_mu_history[self.alice_mu_history[:, 0] ==
                                                frame]
                    print(mus)
                    if len(mus) > 0:
                        self.alice_mu_data.append(mus[0])
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

                if len(self.alice_raw_mu_data) > 0:
                    lmu = np.array(self.alice_raw_mu_data).transpose()
                    interv = (time.time() - self.meas_time, time.time())
                    if interv[0] >= lmu[0][0] - 10 and interv[
                            -1] <= lmu[0][-1] + 10:
                        xs = np.arange(*interv, 1)
                        lmut = np.mean(np.interp(xs, lmu[0], lmu[1]))
                        self.alice_mu_data.append([self.frame, lmut])
                        save_mu(self.frame, lmut,
                                self.folder + "alice_Brightness.csv")
                    else:
                        print("{} is not in [{},{}]".format(
                            interv, lmu[0][0], lmu[0][-1]))

        if (data is not None) and (not self.reset):
            future = self.executor.submit(
                process_data,
                data,
                self.frame,
                self.channels,
                offsets=self.offsets,
                sync_offset=self.sync_offset,
                sent_key=self.sent_key,
                time_filtering=self.time_filtering,
                verbose=True,
                last_valid_sync=self.last_valid_sync,
            )
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
            eval_data, phase_data, click_data, ui_data, sifted_events = res

            for i in range(4):
                future = self.executor.submit(plot_phases, self.canvases[i],
                                              phase_data)
                self.futures.append(future)
            if self.plot_mode == 2:
                future = self.executor.submit(plot_lmu,
                                              self.canvases[4],
                                              click_data,
                                              self.alice_mu_data,
                                              rep_rate=self.reprate,
                                              alice_loss=self.alice_loss,
                                              bob_loss=self.bob_loss)
                self.futures.append(future)
                future = self.executor.submit(plot_lloss,
                                              self.canvases[5],
                                              click_data,
                                              self.alice_mu_data,
                                              rep_rate=self.reprate,
                                              alice_loss=self.alice_loss,
                                              bob_loss=self.bob_loss)
                self.futures.append(future)
                future = self.executor.submit(plot_qber, self.canvases[6],
                                              ui_data)
                self.futures.append(future)
                future = self.executor.submit(plot_skr, self.canvases[7],
                                              ui_data)
                self.futures.append(future)

            for i in range(4):
                if self.plot_mode == 0:
                    future = self.executor.submit(plot_mus,
                                                  self.canvases[i + 4],
                                                  click_data,
                                                  self.floss,
                                                  rep_rate=self.reprate,
                                                  alice_loss=self.alice_loss,
                                                  bob_loss=self.bob_loss)
                elif self.plot_mode == 1:
                    future = self.executor.submit(plot_floss,
                                                  self.canvases[i + 4],
                                                  click_data,
                                                  self.fmu,
                                                  rep_rate=self.reprate,
                                                  alice_loss=self.alice_loss,
                                                  bob_loss=self.bob_loss)

                self.futures.append(future)
            if self.save_sifted:
                future = self.executor.submit(
                    save_sifted, sifted_events,
                    self.folder + "sifted_frame{}.csv".format(frame))
                self.futures.append(future)

            self.offsets, self.sync_offset, self.last_valid_sync = eval_data
            self.signals.new_data.emit(ui_data)
        self.eval_lock.lockForWrite()
        self.last_plotted_frame = frame
        self.eval_lock.unlock()


class Connection(Worker):

    def __init__(self, sock, address, nr=0, parent=None):
        super().__init__()
        print("New Connection from: {}".format(address))
        self.sock = sock
        self.sock.settimeout(1)
        self.address = address
        self.parent = parent
        self.nr = nr

    def receive(self):
        try:
            data = self.recv_msg()
            if data is None:
                raise Exception()
            try:
                data = data.decode("utf-8")
            except:
                data = np.load(BytesIO(data), allow_pickle=True)
            self.signals.new_data.emit((self.nr, data))
        except socket.timeout:
            return
        except Exception as e:
            print(e)
            print("Client " + str(self.address) + " has disconnected")
            self.signals.conn_closed.emit(self.nr)
            self.kill()
            return

    def send(self, message):
        np_bytes = BytesIO()
        np.save(np_bytes, message, allow_pickle=True)
        self.send_msg(np_bytes.getvalue())

    def send_msg(self, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>Q', len(msg)) + msg
        self.sock.sendall(msg)

    def recv_msg(self):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(8)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>Q', raw_msglen)[0]
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def loop(self, i):
        self.receive()


class Server(Worker):

    def __init__(self, host="localhost", port=31415):
        super().__init__()
        self.host = host
        self.port = port
        self.start_server()
        self.conn_nr = 0

    def start_server(self):
        try:
            print("Starting server on: {}:{}".format(self.host, self.port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(10)
            self.sock.settimeout(1)
        except Exception as e:
            print(e)

    def accept(self):
        try:
            sock, address = self.sock.accept()
            conn = Connection(sock, address, nr=self.conn_nr)
            self.signals.new_data.emit(conn)
            self.conn_nr += 1
        except socket.timeout:
            return
        except Exception as e:
            print(e)

    def loop(self, i):
        self.accept()


class Bob_Server(QObject):
    alice_connected = pyqtSignal(bool)
    alice_new_data = pyqtSignal(object)

    def __init__(self, host="localhost", port=31415):
        super().__init__()
        self.server = Server(host, port)
        self.server.signals.new_data.connect(self.new_conn_handler)
        self.threadpool = QThreadPool.globalInstance()
        self.threadpool.start(self.server)
        self.conns = {}
        self.conns_mutex = QMutex()
        self.alice_conn_nr = None

    def new_conn_handler(self, conn):
        conn.signals.new_data.connect(self.handle_hello)
        conn.signals.conn_closed.connect(self.handle_close)
        self.add_conn(conn)
        self.threadpool.start(conn)

    @pyqtSlot(object)
    def handle_hello(self, message):
        conn_nr, msg = message
        if msg == "alice":
            print("Alice says Hello")
            if self.alice_conn_nr == None:
                self.alice_conn_nr = conn_nr
                self.conns[conn_nr].signals.new_data.disconnect(
                    self.handle_hello)
                self.conns[conn_nr].signals.new_data.connect(self.handle_alice)
                self.alice_connected.emit(True)
                return
        print("Hello: {} invalid".format(msg))
        self.conns[conn_nr].kill()

    @pyqtSlot(object)
    def handle_close(self, conn_nr):
        print("Con {} closed".format(conn_nr))
        self.conns[conn_nr].kill()
        self.remove_conn(self.conns[conn_nr])
        if self.alice_conn_nr == conn_nr:
            self.alice_conn_nr = None
            self.alice_connected.emit(False)

    @pyqtSlot(object)
    def handle_alice(self, message):
        conn_nr, msg = message
        #print("Con: {}: Alice: {}".format(conn_nr, msg))
        self.alice_new_data.emit(msg)

    def send_alice(self, message):
        if self.alice_conn_nr is not None:
            self.conns[self.alice_conn_nr].send(message)

    def kill(self):
        for conn in self.conns:
            self.conns[conn].kill()
        self.server.kill()

    def add_conn(self, conn):
        self.conns_mutex.lock()
        self.conns[conn.nr] = conn
        self.conns_mutex.unlock()

    def remove_conn(self, conn):
        self.conns_mutex.lock()
        self.conns.pop(conn.nr)
        self.conns_mutex.unlock()


class Client(Worker):

    def __init__(self, host="localhost", port=31415):
        super().__init__()
        self.host = host
        self.port = port
        self.connected = False

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            conn = Connection(self.sock, "client", nr=0)
            self.connected = True
            self.signals.new_data.emit(conn)
        except:
            self.connected = False
            print("Could not make a connection to the server")

    def loop(self, i):
        if not self.connected:
            self.connect()
        else:
            time.sleep(0.1)


class Alice_Client(QObject):

    def __init__(self, host="localhost", port=31415):
        super().__init__()
        try:
            self.client = Client(host, port)
            self.client.signals.new_data.connect(self.new_conn_handler)
            self.spy = QSignalSpy(self.client.signals.new_data)
            self.name = "alice"
            self.conn = None
            self.threadpool = QThreadPool.globalInstance()
            self.threadpool.start(self.client)
        except Exception as e:
            print(e)

    @pyqtSlot(object)
    def new_conn_handler(self, conn):
        conn.signals.new_data.connect(self.handle_message)
        conn.signals.conn_closed.connect(self.handle_close)
        self.conn = conn
        self.conn.send(self.name)
        self.threadpool.start(conn)

    @pyqtSlot(object)
    def handle_close(self, conn_nr):
        print("Con {} closed".format(conn_nr))
        self.conn = None
        self.client.connected = False

    def kill(self):
        print("kill alice")
        print(self.conn)
        if self.conn is not None:
            self.conn.kill()
        self.client.kill()

    def handle_message(self, message):
        conn_nr, msg = message
        print("{}: {}".format(conn_nr, msg))
