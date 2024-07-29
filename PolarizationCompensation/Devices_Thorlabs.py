import numpy as np
import time
import subprocess

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

    def jog(self, direction, speed):
        input("[Waveplate] Please start Waveplate {}"
              " jog in {} direction with speed {}°/s".format(
                  self.address, direction, speed))

    def stop(self):
        self.stage.stop()

    def home(self):
        print("homing")
        self.stage.home(sync=False)
        self.stage.wait_for_home()
        print("homing complete")


class Waveplates(dev.WAVEPLATES):

    def __init__(self, waveplates):
        self.waveplates = waveplates

    def jog_like_HWP(self, speed=10):
        self.move_to([0, 0, 90])
        print("Setting jog Params")
        self.waveplates[1].stage.setup_jog(step_size=360 / 2,
                                           max_velocity=speed / 2)
        self.waveplates[2].stage.setup_jog(step_size=360, max_velocity=speed)
        print("Start Jog")
        self.waveplates[1].stage.jog("+", kind="builtin")
        self.waveplates[2].stage.jog("+", kind="builtin")

    def move_like_QWP(self, angle):
        self.move_to([135 + angle, 112.5 + angle, 135 + angle])

    def stop(self):
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
        print("Homing Waveplates")
        for wp in self.waveplates:
            wp.stage.home(sync=False)
        print("Waiting for Home")
        for wp in self.waveplates:
            wp.stage.wait_for_home()
        print("Homing complete")


class Alice_LMU(dev.SOURCE):

    def __init__(self, host="local", aliceSettings=None):
        self.commandLine = "alice-control -c {} -b {} -ms {} -md {} -da {} -db {}"
        self.host = host
        if aliceSettings:
            self.aliceSettings = aliceSettings
        else:
            # self.aliceSettings = [[2, 255, 255, 100, 175],
            #                       [2, 206, 255, 100, 175],
            #                       [2, 230, 255, 100, 175],
            #                       [2, 163, 255, 100, 175]]

            self.aliceSettings = [[3, 211, 255, 100, 175],
                                  [2, 236, 255, 100, 175],
                                  [4, 255, 255, 100, 175],
                                  [2, 197, 255, 100, 172]]

    def turn_on(self, pol=None):
        if not pol:
            pol = [1, 2, 3, 4]
        else:
            pol = [pol]
        for p in pol:
            print("Turn on pol: {}".format(p))
            self._send_command(
                self.commandLine.format(p + 1, *self.aliceSettings[p]))

    def turn_off(self):
        print("Turning off Laser")
        self._send_command(self.commandLine.format("-1", *[0, 0, 0, 0, 0]))

    def _send_command(self, command):
        print("sending command:\n{}".format(command))
        if not self.host == "local":
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


class TimeTaggerUltra(dev.TIMESTAMP):

    def __init__(self, channels):
        import TimeTagger

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
