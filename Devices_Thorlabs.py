import numpy as np
import time
import subprocess

from pylablib.devices import Thorlabs
import TimeTagger

import Devices as dev


# Class/Function definition
class K10CR1(dev.WAVEPLATE):

    def __init__(self, address, zero_pos):
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

    def move_to(self, pos):
        self.stage.move_to(pos + self.zero_pos)
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
        self.waveplates[1].stage.setup_jog(step_size=360,
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
        if isinstance(pos, list):
            for i, wp in enumerate(self.waveplates):
                wp.stage.move_to(pos[i])
        else:
            for wp in self.waveplates:
                wp.stage.move_to(pos)

        for wp in self.waveplates:
            wp.stage.wait_move()
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
            self.aliceSettings = [[2, 255, 255, 100, 175],
                                  [2, 255, 255, 100, 175],
                                  [2, 255, 255, 100, 175],
                                  [2, 255, 255, 100, 175]]

    def turn_on(self, pol):
        self._send_command(
            self.commandLine.format(pol, *self.aliceSettings[pol]))

    def turn_off(self):
        self._send_command(self.commandLine.format("-1", *[0, 0, 0, 0, 0]))

    def _send_command(self, command):
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

    def __init__(self):

        self.tt = TimeTagger.createTimeTagger()
        self.tt.setEventDivider(5, 10)
        self.tt.setTriggerLevel(5, 1)
        self.tt.setSoftwareClock(5, 10_000_000)
        self.tt.setTriggerLevel(-1, -0.5)
        self.tt.setTriggerLevel(-2, -0.5)
        self.tt.setTriggerLevel(-3, -0.5)
        self.tt.setTriggerLevel(-4, -0.5)
        time.sleep(5)

    def read(self, t):
        self.stream = TimeTagger.TimeTagStream(self.tt, 1E9, [1, 2, 3, 4])
        self.stream.startFor(t * 1E12)
        print("Measuring the next {}s".format(t))
        self.stream.waitUntilFinished()
        print("done")
        data = self.stream.getData()
        self.stream.stop()
        self.ts = np.array([data.getChannels(), data.getTimestamps()])

    def stop(self):
        self.stream.stop()

    def get_counts_per_second(self):
        bins = np.arange(self.ts[1][0], self.ts[1][-1], 1E12)
        print(bins)
        cps = []
        for i in range(0, 4):
            channel_data = self.ts[1][np.where(self.ts[0] == i + 1)]
            inds = np.digitize(channel_data, bins)
            print(inds)
            _, counts = np.unique(inds, return_counts=True)
            print(counts)
            cps.append(counts)
        cps = np.transpose(np.array(cps))
        print(cps)
        return bins, cps
