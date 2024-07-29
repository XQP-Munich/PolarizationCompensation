#!/usr/bin/env python3
import time, sys, os, shutil
import numpy as np
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
    filename, file_extension = os.path.splitext(file)

    if file_extension == ".txt":
        folder = "./data/measure_qber_" + time.strftime("%Y%m%d-%H%M%S") + "/"
        key_filename = folder + "target_key.txt"
        os.makedirs(folder)
        shutil.copyfile(file, key_filename)
        # Turn on Alice:
        source = devices.Alice_LMU()
        source.turn_on()
        source.send_key(key_filename)

        # Measure clicks using Timetagger:
        timestamp = devices.TimeTaggerUltra(channels)
        ts = timestamp.read(30)
        save_ts(ts, filename + ".bin")
    else:
        ts = load_ts(file)
        key_filename = filename + ".txt"

    # Evaluate
    symbol_map = {k: v["ch"] for k, v in channels.items() if k != "CLK"}
    inv_symbol_map = {v: k for k, v in symbol_map.items()}

    target_key = []
    with open(key_filename, "r") as file:
        for char in file.readline()[:-1]:
            target_key.append(symbol_map[char] - 1)

    dt = 10000
    key_length = len(target_key)
    offset = 0
    chan_offset = [1000, 1000, -5500, 2100, 0]
    raw_data = ts.view(np.int64)  #[:,int(len(ts/2)):]
    raw_data[1] = raw_data[1]
    sync_data = raw_data[1][raw_data[0] == channels["SYNC"]["ch"]]
    meas_time = raw_data[1][-1] - raw_data[1][0]

    print(f"Got {len(raw_data[0])} Events in {meas_time*1E-12}s")

    if sync_data.size != 0:
        print("Sync Chan detected")
        raw_data[1] -= sync_data[1]
        raw_data = raw_data[:, raw_data[1] >= 0]
        meas_time = raw_data[1][-1] - raw_data[1][0]
        print(
            f"Got {len(raw_data[0])} Events in {meas_time*1E-12}s after first SYNC"
        )

    offsets = []
    for i in range(0, 4):
        filter_var = 10000
        offset = -2000
        while filter_var > 2000:
            offset += 2000
            if offset >= 10000:
                print(f"max{i}")
                break
            channel_data = raw_data[1][raw_data[0] == i +
                                       1] + offset + chan_offset[i]
            time_in_symbol = (channel_data % dt)
            filter_var = np.std(time_in_symbol)
        offsets.append(offset)
    offset = max(offsets)
    print(f"Setting offset to: {offset}")

    filter_means = []
    filter_vars = []
    for i in range(0, 4):
        channel_data = raw_data[1][raw_data[0] == i +
                                   1] + offset + chan_offset[i]
        time_in_symbol = (channel_data % dt)
        filter_means.append(np.median(time_in_symbol))
        filter_vars.append(np.std(time_in_symbol))
    filter_mean = np.mean(filter_means)
    filter_var = np.mean(filter_vars)

    filter_min = filter_mean - 0.55 * filter_var
    filter_max = filter_mean + 0.65 * filter_var
    print("Setting filter from {:.0f}ps to {:.0f}ps".format(
        filter_min, filter_max))

    data = raw_data.copy()
    for i in range(0, 5):
        data[1, data[0] == i + 1] += offset + chan_offset[i]

    phases = data[1] % dt
    plt.figure(figsize=(5, 3), dpi=400)
    nmax = []
    for i in range(0, 5):
        n, _, _ = plt.hist(phases[data[0] == i + 1],
                           bins=100,
                           alpha=0.4,
                           label=f"{inv_symbol_map[i+1]}")
        nmax.append(np.max(n))
    plt.vlines(filter_min, 0, max(nmax))
    plt.vlines(filter_max, 0, max(nmax))

    time_mask = np.logical_or(
        np.logical_and(phases >= filter_min, phases <= filter_max),
        data[0] == 5)
    data = data[:, time_mask]

    bins = np.zeros((key_length, 4))
    for i in range(0, 4):
        pos_in_key = ((data[1, data[0] == i + 1] % (dt * key_length)) /
                      dt).astype(int)
        unique_indices, counts = np.unique(pos_in_key, return_counts=True)
        for id, cnt in zip(unique_indices, counts):
            bins[id][i] += cnt

    keys_sent = meas_time / (key_length * dt)

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

    for i in range(4):
        probs = det_probs[det_key == i]
        print("Detection Prob for {}: {:.6f}% with a std of {:.8f}".format(
            inv_symbol_map[i + 1],
            np.mean(probs) * 100,
            np.std(probs) * 100))

    print("Mean Qber: {:.2f}% with std of {:.4f}".format(
        np.mean(qbers) * 100,
        np.std(qbers) * 100))
    print("Max Qber: {:.2f}% at {}".format(
        np.max(qbers) * 100, bins[np.argmax(qbers)]))
    print("Sifted key Rate: {:.2f}".format(num_sifted_det / meas_time / 1E-12))
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
            np.mean(qbers) * 100, num_sifted_det / meas_time / 1E-12 / 1000))
    plt.xlabel("Detection time in ps")
    plt.ylabel("Number of detections")
    plt.tight_layout()

    if (max(sames) == len(target_key)):
        sync_phases = data[1][data[0] == 5] % dt
        if len(sync_phases) > 0:
            print(
                "Detected key matches sent key with offset: {} and phase: {} std {}"
                .format(np.argmax(sames), np.mean(sync_phases),
                        np.std(sync_phases)))
        else:
            print("Detected key matches sent key with offset: {}".format(
                np.argmax(sames)))
        plt.savefig(filename + ".png")

    else:
        print(len(sames), max(sames), np.argmax(sames))

        shift_key = temp_key[np.argmax(sames):np.argmax(sames) +
                             len(target_key)]
        sumcounts = [[], [], [], []]
        for i in range(len(target_key)):
            if (key[i] != shift_key[i]):
                print(i, key[i], shift_key[i], bins[i], np.sum(bins[i]))
            sumcounts[key[i]].append(np.sum(bins[i]))
        for i in range(4):
            print(i, np.mean(sumcounts[i]))
        plt.show()
