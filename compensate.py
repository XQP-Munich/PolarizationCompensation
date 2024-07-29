#!/usr/bin/env python3

import sys, os, time
import numpy as np
import PolarizationCompensation.Measurements as mm
import PolarizationCompensation.Devices_Thorlabs as devices

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
        qwp2 = devices.K10CR1(55232924, 308.24)
        hwp = devices.K10CR1(55232454, 231.59)
        qwp1 = devices.K10CR1(55232464, 301.01)

        waveplates = devices.Waveplates([qwp1, hwp, qwp2])
        source = devices.Alice_LMU()
        timestamp = devices.TimeTaggerUltra(channels)

        # Do this every time it is moved
        # Send any Linear Pos into bob

        _, counts = mm.measure_Coupling_Efficiency_cont(
            source, waveplates, timestamp)
        np.save(path + "counts_coupling.npy", counts)
    else:
        counts = np.load(path + "counts_coupling.npy")
    coupling = mm.evaluate_Coupling_Efficiency(counts, inv_symbol_map, path)
    print("Coupling:\n", coupling)

    send_pol = ["V", "M"]
    waveplate_setup = ["QWP", "HWP", "QWP"]
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
