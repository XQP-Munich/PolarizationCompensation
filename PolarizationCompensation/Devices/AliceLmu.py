import subprocess
import os
import numpy as np

from PolarizationCompensation.Devices.Templates import SOURCE


class AliceLmu(SOURCE):

    def __init__(self, host="local", aliceSettings=None):
        self.commandLine = "alice-control -c {} -b {} -ms {} -md {} -da {} -db {}"
        self.host = host
        self.channel_map = {"H": 1, "V": 2, "P": 3, "M": 4}
        if aliceSettings:
            self.aliceSettings = aliceSettings
        else:
            self.aliceSettings = {
                1: {
                    "H": [3, 255, 186, 100, 100 + 76],
                    "V": [3, 204, 197, 96, 96 + 70],
                    "P": [4, 237, 172, 117, 117 + 66],
                    "M": [3, 180, 176, 82, 82 + 63]
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
        dir, filename = os.path.split(filename)
        filename, file_extension = os.path.splitext(filename)

        keyfilename = dir + "/" + filename + ".key"
        data.tofile(keyfilename)
        if not self.host == "local":
            self._send_command("scp {} {}:~/{}".format(
                os.path.abspath(keyfilename), self.host, filename + ".key"),
                               forcelocal=True)
            return "~/" + filename + ".key"
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
