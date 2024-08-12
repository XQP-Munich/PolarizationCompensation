#!/usr/bin/env python3

import time, sys, os, shutil
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import PolarizationCompensation.Devices_Thorlabs as devices

# setup variables

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
    "SYNC": {
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


def save_ts(ts, path):
    data = np.bitwise_or(ts[0], np.left_shift(ts[1], 4))
    data.tofile(path)


def load_ts(path):
    data = np.fromfile(path, dtype=np.uint64)
    return np.array([np.bitwise_and(data, 15), np.right_shift(data, 4)])


if __name__ == "__main__":
    folder = "./data/measure_qber_" + time.strftime("%Y%m%d-%H%M%S") + "/"
    file = sys.argv[1]
    if not os.path.exists(file):
        print("Cannot open {}".format(file))
        sys.exit()

    filename, file_extension = os.path.splitext(os.path.basename(file))

    if file_extension == ".txt":
        filename = folder + filename
        folder = "./data/measure_qber_" + time.strftime("%Y%m%d-%H%M%S") + "/"
        key_filename = filename + file_extension
        os.makedirs(folder)
        shutil.copyfile(file, key_filename)
        # Turn on Alice:
        source = devices.Alice_LMU(host="t14s")
        source.turn_on()
        source.send_key(key_filename)

        # Measure clicks using Timetagger:
        timestamp = devices.TimeTaggerUltra(channels)
        ts = timestamp.read(120)
        save_ts(ts, filename + ".bin")
        source.stop_key()
    elif file_extension == ".ttbin":
        # Measure clicks using Virtual Timetagger:
        timestamp = devices.TimeTaggerUltraVirtual(channels, file)
        ts = timestamp.read(10)
        save_ts(ts, filename + ".bin")
        filename, file_extension = os.path.splitext(file)
        key_filename = filename + ".txt"
    else:
        ts = load_ts(file)
        filename, file_extension = os.path.splitext(file)
        key_filename = filename + ".txt"

    # Evaluate
    symbol_map = {k: v["ch"] for k, v in channels.items() if k != "CLK"}
    inv_symbol_map = {v: k for k, v in symbol_map.items()}

    target_key = []
    with open(key_filename, "r") as file:
        for char in file.readline()[:-1]:
            target_key.append(symbol_map[char.upper()] - 1)

    dt = 10000
    key_length = len(target_key)
    print(key_length)
    bob_eff = 25.3852 / 100
    filter = 1.32 / 100
    rep_rate = 1 / (dt * 1E-12)
    alice_trans = 0.917
    offset = 0
    chan_offset = [1000, 1071, -5515, 2070, 0]
    raw_data = ts.view(np.int64)  # [:,int(len(ts/2)):]
    raw_data[1] = raw_data[1]
    sync_data = raw_data[1][raw_data[0] == channels["SYNC"]["ch"]]
    meas_time = (raw_data[1][-1] - raw_data[1][0]) * 1E-12

    if sync_data.size != 0:
        print("Sync Chan detected")
        raw_data[1] -= sync_data[1]
        raw_data = raw_data[:, raw_data[1] >= 0]
        meas_time = (raw_data[1][-1] - raw_data[1][0]) * 1E-12
        print("Got {:.2e} Events in {:.0f}s after first SYNC".format(
            len(raw_data[0]), meas_time))

        offset = 52000
        print("Setting offset to: {}ps".format(offset))
    else:
        print("Got {:.2e} Events in {:.0f}s".format(len(raw_data[0]),
                                                    meas_time))
        offsets = []
        for i in range(0, 4):
            filter_var = 10000
            offset = -2000
            while filter_var > 2000:
                offset += 2000
                if offset >= 10000:
                    print("max offset for channel {}".format(
                        inv_symbol_map[i + 1]))
                    break
                channel_data = raw_data[1][raw_data[0] == i +
                                           1] + offset + chan_offset[i]
                time_in_symbol = (channel_data % dt)
                filter_var = np.std(time_in_symbol)
            offsets.append(offset)
        offset = max(offsets)
        offset = 265000
        print("Setting offset to: {}ps".format(offset))

    data = raw_data.copy()
    for i in range(0, 5):
        data[1, data[0] == i + 1] += offset + chan_offset[i]

    bin_width = 100
    bins = np.histogram(data[1, data[0] != channels["SYNC"]["ch"]] % dt,
                        bins=bin_width)[0]
    peaks, _ = sp.signal.find_peaks(bins, height=max(bins) / 2)
    _, _, filter_min, filter_max = sp.signal.peak_widths(bins,
                                                         peaks,
                                                         rel_height=0.7)
    filter_min = filter_min[0] * bin_width
    filter_max = filter_max[0] * bin_width

    print("Setting filter from {:.0f}ps to {:.0f}ps".format(
        filter_min, filter_max))

    phases = data[1] % dt
    plt.figure(figsize=(5, 3), dpi=400)
    nmax = []
    for i in range(0, 4):
        n, _, _ = plt.hist(phases[data[0] == i + 1],
                           bins=100,
                           alpha=0.4,
                           label=inv_symbol_map[i + 1])
        nmax.append(np.max(n))
    phase = phases[data[0] == channels["SYNC"]["ch"]]
    n, _ = np.histogram(phase, bins=100, range=(0, dt))
    weight = max(nmax) / max(n)
    plt.hist(phase,
             bins=100,
             range=(0, dt),
             alpha=0.4,
             label="SYNC*",
             weights=np.ones(len(phase)) * weight)

    plt.vlines(filter_min, 0, max(nmax))
    plt.vlines(filter_max, 0, max(nmax))

    # plt.figure()
    # pos_in_key = np.array(((data[1, :] % (dt * key_length)) /
    #                   dt).astype(int))
    # phases_key = [[],[],[],[]]
    # for i in range(key_length):
    #     ph=phases[np.logical_and(pos_in_key==i,data[0] == 4)]
    #     phases_key[target_key[i]] = phases_key[target_key[i]]+ph.tolist()
    # ms=[]
    # for i in range(4):
    #     m=np.mean(phases_key[i])
    #     n, _, _ = plt.hist(phases_key[i],
    #                        bins=1000,
    #                        alpha=0.4,
    #                        label=inv_symbol_map[i + 1])
    #     plt.vlines(m,0,max(n))
    #     ms.append(m)
    # print(ms)
    # for i in range(4):
    #     print("Shift {} by {}".format(inv_symbol_map[i + 1],ms[i]/4.45))
    # plt.legend()
    # plt.figure()
    # for i in range(4):
    #     ph=np.array(phases_key[i])-ms[i]+2000
    #     m=np.mean(ph)
    #     n, _, _ = plt.hist(ph,
    #                        bins=1000,
    #                        alpha=0.4,
    #                        label=inv_symbol_map[i + 1])
    #     plt.vlines(m,0,max(n))
    # plt.legend()

    # plt.show()

    time_mask = np.logical_or(
        np.logical_and(phases >= filter_min, phases <= filter_max),
        data[0] == 5)
    data = data[:, time_mask]

    for i in range(0, 5):
        print("Got {} {} Events".format(len(data[1, data[0] == i + 1]),
                                        inv_symbol_map[i + 1]))

    bins = np.zeros((key_length, 4))
    for i in range(0, 4):
        pos_in_key = ((data[1, data[0] == i + 1] % (dt * key_length)) /
                      dt).astype(int)
        unique_indices, counts = np.unique(pos_in_key, return_counts=True)
        for id, cnt in zip(unique_indices, counts):
            bins[id][i] += cnt

    keys_sent = meas_time / (key_length * dt * 1E-12)

    key = []
    qbers = []
    det_probs = []
    num_sifted_det = 0
    for i in range(key_length):
        count = bins[i]
        diff1 = count[0] - count[1]
        diff2 = count[2] - count[3]
        diff3 = abs(diff1) - abs(diff2)
        val = 0
        qber = 0
        if diff3 > 0:
            if diff1 > 0:
                val = 0
                qber = count[1] / (count[0] + count[1])
                det_prob = count[0] / keys_sent
            else:
                val = 1
                qber = count[0] / (count[0] + count[1])
                det_prob = count[1] / keys_sent
            num_sifted_det += count[0] + count[1]
        else:
            if diff2 > 0:
                val = 2
                qber = count[3] / (count[2] + count[3])
                det_prob = count[2] / keys_sent
            else:
                val = 3
                qber = count[2] / (count[2] + count[3])
                det_prob = count[3] / keys_sent
            num_sifted_det += count[2] + count[3]
        key.append(val)
        qbers.append(qber)
        det_probs.append(det_prob)

    det_probs = np.array(det_probs)
    det_key = np.array(key)
    det_probs_pol = []
    for i in range(4):
        probs = det_probs[det_key == i]
        det_probs_pol.append([probs.mean(), probs.std()])
    det_probs_pol = np.array(det_probs_pol)
    for i in range(4):
        print(
            "Detection Prob for {}: {:.6f}% with a std of {:.8f} rel: {:.2f}".
            format(inv_symbol_map[i + 1], det_probs_pol[i][0] * 100,
                   det_probs_pol[i][1] * 100,
                   det_probs_pol[i][0] / np.max(det_probs_pol[:, 0])))

    print("Mean Qber: {:.2f}% with std of {:.4f}".format(
        np.nanmean(qbers) * 100,
        np.nanstd(qbers) * 100))
    print("Max Qber: {:.2f}% at {}".format(
        np.max(qbers) * 100, bins[np.argmax(qbers)]))
    print("Sifted key Rate: {:.2f}".format(num_sifted_det / meas_time))
    temp_key = target_key + target_key
    index_offset = 0
    sames = []
    for i in range(len(target_key)):
        comp_key = temp_key[i:i + len(target_key)]
        same = 0
        for j in range(len(target_key)):
            if (key[j] == comp_key[j]):
                same += 1
        sames.append(same)

    plt.legend()
    plt.title(
        "Pulses with mean QBER={:.2f}%\nSifted key rate={:.2f}Kb/s".format(
            np.nanmean(qbers) * 100, num_sifted_det / meas_time / 1000))
    plt.xlabel("Detection time in ps")
    plt.ylabel("Number of detections")
    plt.tight_layout()

    if (max(sames) == len(target_key)):
        sync_phases = data[1][data[0] == 5] % dt
        ind_offset = np.argmax(sames)
        if len(sync_phases) > 0:
            print("Keys match with offset: {} and phase: {:.2f} std {:.4f}".
                  format(ind_offset, np.mean(sync_phases),
                         np.std(sync_phases)))
        else:
            print("Detected key matches sent key with offset: {}".format(
                np.argmax(sames)))
        plt.savefig(filename + ".png")

        key = np.array(temp_key[ind_offset:ind_offset + len(target_key)])
        for i in range(4):
            pol_bin = bins[key == i, :]
            mu = pol_bin.sum() / meas_time / bob_eff / filter / rep_rate / (
                len(pol_bin) / key_length) * alice_trans
            print("Mean Photon Numbers for {}: {:.3f}".format(
                inv_symbol_map[i + 1], mu))

    else:
        print(len(sames), max(sames), np.argmax(sames))
        print(bins[10])
        print(bins[11])
        print(bins[12])

        shift_key = temp_key[np.argmax(sames):np.argmax(sames) +
                             len(target_key)]
        sumcounts = [[], [], [], []]
        for i in range(len(target_key)):
            if (key[i] != shift_key[i]):
                print(i, key[i], shift_key[i], bins[i], np.sum(bins[i]))
            sumcounts[key[i]].append(np.sum(bins[i]))
        for i in range(4):
            print(i, np.mean(sumcounts[i]))
        plt.savefig(filename + ".png")
        plt.show()
