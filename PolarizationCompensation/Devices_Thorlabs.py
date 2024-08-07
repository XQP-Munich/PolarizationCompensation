import numpy as np
import time
import subprocess
import os

import PolarizationCompensation.Devices as dev


# Class/Function definition
class K10CR1(dev.WAVEPLATE):

    def __init__(self, address, zero_pos):
        from pylablib.devices import Thorlabs
        self.address = address
        self.zero_pos = zero_pos
        print("Looking for Motor with id={}".format(address))
        for i in range(20):
            try:
                conn = {
                    "port": "/dev/ttyUSB{}".format(i),
                    "baudrate": 115200,
                    "rtscts": True
                }
                self.stage = Thorlabs.KinesisMotor(("serial", conn),
                                                   scale="stage")
                if (self.stage.get_device_info()[0] == int(address)):
                    print("Found Motor with id={}".format(address))
                    break
            except:
                print("No Device at /dev/ttyUSB{}".format(i))

    def move_to(self, pos, block=True):
        self.stage.move_to(pos + self.zero_pos)
        if block:
            self.wait_move()

    def wait_move(self):
        self.stage.wait_move()

    def stop(self):
        self.stage.stop()

    def home(self, block=True):
        print("Homing")
        self.stage.home(sync=False)
        if block:
            self.wait_home()

    def wait_home(self):
        self.stage.wait_for_home()
        print("Homing complete")

    def setup_jog(self, speed, limit=720):
        self.stage.setup_jog(step_size=limit, max_velocity=speed)

    def jog(self, direction="+"):
        self.stage.jog(direction, kind="builtin")


class Waveplates(dev.WAVEPLATES):

    def __init__(self, waveplates, hwp_angle0, hwp_speed, qwp_angle0):
        self.waveplates = waveplates
        self.hwp_angle0 = hwp_angle0
        self.hwp_speed = hwp_speed
        self.qwp_angle0 = qwp_angle0

    def jog_like_HWP(self, speed):
        self.move_to(self.hwp_angle0)
        print("Setting jog params")
        for i, wp in enumerate(self.waveplates):
            wp.setup_jog(speed=self.hwp_speed[i] * speed)
        print("Start jog")
        for wp in self.waveplates:
            wp.jog()

    def move_like_QWP(self, angle):
        self.move_to([x + angle for x in self.qwp_angle0])

    def stop(self):
        print("Stopping waveplates")
        for wp in self.waveplates:
            wp.stop()

    def move_to(self, pos):
        print("Moving Waveplates to {}".format(pos))
        if isinstance(pos, list) or isinstance(pos, np.ndarray):
            for i, wp in enumerate(self.waveplates):
                wp.move_to(pos[i], block=False)
        else:
            for wp in self.waveplates:
                wp.move_to(pos, block=False)

        for wp in self.waveplates:
            wp.wait_move()
        print("done")

    def home(self):
        for wp in self.waveplates:
            wp.home(block=False)
        print("Waiting for home")
        for wp in self.waveplates:
            wp.wait_home()


class Alice_LMU(dev.SOURCE):

    def __init__(self, host="local", aliceSettings=None):
        self.commandLine = "alice-control -c {} -b {} -ms {} -md {} -da {} -db {}"
        self.host = host
        self.channel_map = {"H": 1, "V": 2, "P": 3, "M": 4}
        if aliceSettings:
            self.aliceSettings = aliceSettings
        else:
            # self.aliceSettings = [[2, 255, 255, 100, 175],
            #                       [2, 206, 255, 100, 175],
            #                       [2, 230, 255, 100, 175],
            #                       [2, 163, 255, 100, 175]]

            self.aliceSettings = {
                1 : {
                    "H": [3, 228, 255, 100, 175],
                    "V": [2, 233, 255, 100, 175],
                    "P": [4, 255, 255, 100, 175],
                    "M": [2, 205, 255, 100, 173]
                },
                2 : {
                    "H": [3, 228, 255, 100, 175],
                    "V": [2, 233, 255, 100, 175],
                    "P": [4, 255, 255, 100, 175],
                    "M": [2, 200, 255, 100, 172]
                }
            }

    def turn_on(self, pol=None, set=1):
        if not pol:
            pol = ["H", "V", "P", "M"]
        else:
            pol = [pol]
        for p in pol:
            print("Turn on pol: {}".format(p))
            self._send_command(
                self.commandLine.format(self.channel_map[p],
                                        *self.aliceSettings[set][p]))

    def turn_off(self):
        print("Turning off Laser")
        self._send_command(self.commandLine.format("-1", *[0, 0, 0, 0, 0]))

    def _send_command(self, command, forcelocal=False):
        print("sending command:\n{}".format(command))
        if not (self.host == "local" or forcelocal):
            return self._send_command_ssh(command)

        proc = subprocess.Popen(command.split(" "),
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        output, error = proc.communicate()

        if error:
            raise Exception(f"Error {error.decode('utf-8')}")

        return output.decode('utf-8')

    def _send_command_ssh(self, command):

        ssh = subprocess.Popen(["ssh", self.host, command],
                               shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        output, error = ssh.communicate()

        if error:
            raise Exception(f"Error {error.decode('utf-8')}")

        return output.decode('utf-8')

    def txt_to_key(self, filename):
        data = []
        conv = {"H": 2, "V": 3, "P": 6, "M": 7, "h": 0, "v": 1, "p": 4, "m": 5}
        with open(filename, "r") as file:
            for line in file.readlines():
                line = line.rstrip("\n")
                print(len(line))
                first = True
                byte = np.ubyte(0)
                for char in line:
                    if first:
                        byte = np.ubyte(conv[char])
                    else:
                        byte = np.bitwise_or(byte, np.ubyte(16 * conv[char]))
                        data.append(byte)
                    first = not first

        data = np.array(data)
        dir,filename=os.path.split(filename)
        filename, file_extension = os.path.splitext(filename)

        keyfilename = dir +"/"+ filename + ".key"
        data.tofile(keyfilename)
        if not self.host == "local":
            time.sleep(1)
            self._send_command("scp {} {}:~/{}".format(os.path.abspath(keyfilename),self.host,filename+".key"), forcelocal=True)
            return "~/"+ filename + ".key"
        return filename

    def send_key(self, key):
        filename, file_extension = os.path.splitext(key)
        if file_extension == ".key":
            self._send_command("ram-playback -if {}".format(key))
        else:
            self._send_command("ram-playback -if {}".format(
                self.txt_to_key(key)))
            
    def stop_key(self):
        self._send_command("ram-playback -s")



class TimeTaggerUltra(dev.TIMESTAMP):

    def __init__(self, channels):
        import TimeTagger
        print("Initializing timetagger")

        self.channel_dict = channels
        self.channels_measure = [
            channels[c]["ch"] for c in channels if c != "CLK"
        ]
        self.tt = TimeTagger.createTimeTagger()
        for key in channels:
            self.tt.setTriggerLevel(
                channels[key]["ch"] * channels[key]["edge"],
                channels[key]["trigger"])
            if key == "CLK":
                self.tt.setEventDivider(channels[key]["ch"], 1)
                self.tt.setSoftwareClock(channels[key]["ch"], 10_000_000)

        time.sleep(5)

    def read(self, t):
        import TimeTagger

        try_count = 0
        while True:
            try_count += 1
            self.stream = TimeTagger.TimeTagStream(self.tt, 1E9,
                                                   self.channels_measure)
            self.stream.startFor(t * 1E12)
            print("Measuring the next {}s".format(t))
            self.stream.waitUntilFinished()
            print("Done")
            data = self.stream.getData()
            self.stop()
            scs = self.tt.getSoftwareClockState()
            if scs.error_counter == 0:
                self.ts = np.array([data.getChannels(), data.getTimestamps()])
                return self.ts
            print("Clock errors, trying again!")
            time.sleep(5)
            if try_count == 5:
                Exception("To many clock erros")

    def stop(self):
        self.stream.stop()

    def get_counts_per_second(self):
        bins = np.arange(self.ts[1][0], self.ts[1][-1], 1E12)
        cps = []
        for i in range(0, 4):
            channel_data = self.ts[1][np.where(self.ts[0] == i + 1)]
            inds = np.digitize(channel_data, bins)
            _, counts = np.unique(inds, return_counts=True)
            cps.append(counts)
        cps = np.transpose(np.array(cps))
        return bins, cps
