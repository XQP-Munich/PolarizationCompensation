import subprocess
import os
import numpy as np
import time

from Devices.Templates import TIMESTAMP


class TimeTaggerLmu():

    def __init__(self, host="local", simulate=False):
        self.commandLine = "counter -t 1 -s {}"
        self.host = host
        self.simulate = simulate

    def get_counts_per_second(self, t):
        if self.simulate:
            return 10000 + 0 * np.random.random()
        cnts = []
        data = self._send_command(self.commandLine.format(t))
        for line in data.split("\n"):
            if line != "":
                sp = [x for x in line.split(" ") if x != ""]
                if len(sp) != 10:
                    print(line)
                cnts.append(int(sp[2]))

        return np.mean(cnts)

    def _send_command(self, command):
        print("sending command:\n{}".format(command))
        if not (self.host == "local"):
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
