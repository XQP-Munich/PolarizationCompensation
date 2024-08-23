#!/usr/bin/env python3

import sys, os, time
import numpy as np
import PolarizationCompensation.Measurements as mm
from Devices.K10CR1 import K10CR1
from Devices.Waveplates import Waveplates
from Devices.AliceLmu import AliceLmu
from Devices.TimeTaggerUltra import TimeTaggerUltra

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
    "SYC": {
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

send_pol = ["V", "P"]
waveplate_setup = ["QWP", "HWP", "QWP"]
aliceSettings = {
    "H": [0, 0, 0, 100, 175],
    "V": [14, 0, 0, 100, 175],
    "P": [0, 0, 0, 100, 175],
    "M": [15, 0, 0, 100, 172]
}

if __name__ == "__main__":
    symbol_map = {k: v["ch"] - 1 for k, v in channels.items() if k != "CLK"}
    inv_symbol_map = {v: k for k, v in symbol_map.items()}

    path = "./data/polcomp_" + time.strftime("%Y%m%d-%H%M%S") + "/"
    if len(sys.argv) == 2:
        path = sys.argv[1] + "/"

    folder_exists = os.path.exists(path)
    if not folder_exists:
        os.makedirs(path)

        # initialize devices
        qwp2 = K10CR1(55232924, 308.24)
        hwp = K10CR1(55232454, 231.59)
        qwp1 = K10CR1(55232464, 301.01)

        # get values of hwp_angle0,hwp_speed,qwp_angle from simulation script
        waveplates = Waveplates([qwp1, hwp, qwp2],
                                hwp_angle0=[0, 0, 90],
                                hwp_speed=[1, 1, 1],
                                qwp_angle0=[135, 112.5, 135])
        source = AliceLmu("t14s")
        timestamp = TimeTaggerUltra(channels)

        # Do this every time it is moved
        # Send any Linear Pos into bob

        _, counts = mm.measure_Coupling_Efficiency_cont(
            source, waveplates, timestamp)
        np.save(path + "counts_coupling.npy", counts)
        #coupling = [0.50516748, 0.49483252, 0.49319967, 0.50680033]
    else:
        counts = np.load(path + "counts_coupling.npy")
        #coupling = [0.50516748, 0.49483252, 0.49319967, 0.50680033]

    coupling = mm.evaluate_Coupling_Efficiency(counts, inv_symbol_map, path)
    print("Coupling:\n", coupling)

    if not folder_exists:
        # Do this every Time
        bins, counts = mm.measure_Polcomp_Counts(source, waveplates, timestamp,
                                                 send_pol)
        np.save(path + "counts_polcomp.npy", counts)
    else:
        counts = np.load(path + "counts_polcomp.npy")

    angles, loss, stokesangles = mm.evaluate_Polcomp_Counts(
        counts, coupling, send_pol, waveplate_setup)
    print("Compensation angles are:\n{}\nwith a loss of {}".format(
        angles, loss))

    if not folder_exists:
        print("Moving waveplats to compensation angle")
        waveplates.move_to(angles)
