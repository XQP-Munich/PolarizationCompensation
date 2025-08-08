import subprocess
import os
import numpy as np
import time
import re
from Devices.Templates import TIMESTAMP


class TimeTaggerLmu(TIMESTAMP):
    def __init__(self, channels):
        self.channels = channels
        result = subprocess.run(
            ["counter", "-t", "0.1", "-s", "1"], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise Exception("Error:\n", result.stderr)

    def read(self, t):
        return super().read(t)

    def stop(self):
        return super().stop()

    def get_counts_per_second(self, t):
        result = subprocess.run(
            ["counter", "-t", "1", "-s", str(t)], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise Exception("Error:\n", result.stderr)
        data = np.array(
            [
                list(map(int, re.split(r"\s+", line.strip())))
                for line in result.stdout.strip().splitlines()
            ]
        )
        data_filtered = data[
            :,
            [
                self.channels["H"]["ch"] + 2,
                self.channels["V"]["ch"] + 2,
                self.channels["P"]["ch"] + 2,
                self.channels["M"]["ch"] + 2,
            ],
        ]
        return data[:, 0] * 1e-12, data_filtered
