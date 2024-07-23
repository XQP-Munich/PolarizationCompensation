import TimeTagger
import time
import numpy as np

while True:
    with TimeTagger.createTimeTagger() as tt:
        tt.setEventDivider(5, 5)
        tt.setTriggerLevel(5, 1)
        tt.setSoftwareClock(5, 10_000_000)
        tt.setTriggerLevel(-1, -0.5)
        tt.setTriggerLevel(-2, -0.5)
        tt.setTriggerLevel(-3, -0.5)
        tt.setTriggerLevel(-4, -0.5)
        time.sleep(5)
        stream = TimeTagger.TimeTagStream(tt, 1E9, [1, 2, 3, 4])
        stream.startFor(30 * 1E12)
        stream.waitUntilFinished()
        data = stream.getData()
        stream.stop()
        scs = tt.getSoftwareClockState()
        ts = np.array([data.getChannels(), data.getTimestamps()])
        if scs.error_counter == 0:
            break
    time.sleep(5)

import numpy as np
import matplotlib.pyplot as plt

target_key = [0]
symbol_map = {"H": 0, "V": 1, "P": 2, "M": 3}
inv_symbol_map = {v: k for k, v in symbol_map.items()}
with open("targetKey_QUBEmeasurements.txt", "r") as file:
    for char in file.readline()[:-1]:
        target_key.append(symbol_map[char])

dt = 10000
key_length = 2049
offset = 0
chan_offset = [1000, 1000, -5500, 2100]
#raw_data = np.reshape(np.fromfile("./data/data2.bin", dtype=np.int64), (2, -1))
raw_data = ts  #[:,int(len(ts/2)):]
meas_time = raw_data[1][-1] - raw_data[1][0]

#raw_data = raw_data[:, :10]
sorted_data = []
bins = np.zeros((key_length, 4))

print(f"Got {len(raw_data[0])} Events in {meas_time*1E-12}s")

offsets = []
for i in range(0, 4):
    filter_var = 10000
    offset = -2000
    while filter_var > 2000:
        offset += 2000
        if offset >= 10000:
            print(f"max{i}")
            break
        channel_data = raw_data[1][np.where(raw_data[0] == i +
                                            1)] + offset + chan_offset[i]
        time_in_symbol = (channel_data % dt)
        filter_var = np.std(time_in_symbol)
    offsets.append(offset)
offset = max(offsets)
print(f"Setting offset to: {offset}")

filter_means = []
filter_vars = []
for i in range(0, 4):
    channel_data = raw_data[1][np.where(raw_data[0] == i +
                                        1)] + offset + chan_offset[i]
    time_in_symbol = (channel_data % dt)
    filter_means.append(np.median(time_in_symbol))
    filter_vars.append(np.std(time_in_symbol))
filter_mean = np.mean(filter_means)
filter_var = np.mean(filter_vars)

filter_min = filter_mean - 0.35 * filter_var
filter_max = filter_mean + 0.45 * filter_var
print("Setting filter from {:.0f}ps to {:.0f}ps".format(
    filter_min, filter_max))

for i in range(0, 4):
    channel_data = raw_data[1][np.where(raw_data[0] == i +
                                        1)] + offset + chan_offset[i]
    time_in_symbol = (channel_data % dt)

    phase = channel_data % dt
    plt.hist(phase, bins=100, alpha=0.4, label=f"{inv_symbol_map[i]}")

    channel_data_filtered = channel_data[np.where(
        np.logical_and(time_in_symbol >= filter_min, time_in_symbol
                       <= filter_max))]
    #phase_filtered = channel_data_filtered % dt
    #hist_bins = int((filter[1] - filter[0]) / 100)
    #plt.hist(phase_filtered, bins=20, alpha=0.6, label=f"{inv_symbol_map[i]}")
    plt.vlines([filter_min, filter_max], 0, 10000)

    pos_in_key = ((channel_data_filtered % (dt * key_length)) / dt).astype(int)
    unique_indices, counts = np.unique(pos_in_key, return_counts=True)
    for id, cnt in zip(unique_indices, counts):
        bins[id][i] += cnt

keys_sent = meas_time / (key_length * dt)

key = []
qbers = []
det_probs = []
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
    else:
        if diff2 > 0:
            val = 2
            qber = count[3] / (count[2] + count[3])
            det_prob = count[2] / keys_sent
        else:
            val = 3
            qber = count[2] / (count[2] + count[3])
            det_prob = count[3] / keys_sent
    key.append(val)
    qbers.append(qber)
    det_probs.append(det_prob)

det_probs = np.array(det_probs)
det_key = np.array(key)

for i in range(4):
    probs = det_probs[np.where(det_key == i)]
    print("Detection Prob for {}: {:.6f}% with a std of {:.8f}".format(
        inv_symbol_map[i],
        np.mean(probs) * 100,
        np.std(probs) * 100))

#print(bins)
#print(keyr)
#print(qbers)
print("Mean Qber: {:.2f}% with std of {:.4f}".format(
    np.mean(qbers) * 100,
    np.std(qbers) * 100))
print("Max Qber: {:.2f}% at {}".format(
    np.max(qbers) * 100, bins[np.argmax(qbers)]))
plt.legend()
plt.show()
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
if (max(sames) == len(target_key)):
    print("Detected key matches sent key!!")
else:
    print(len(sames), max(sames), np.argmax(sames))

    shift_key = temp_key[np.argmax(sames):np.argmax(sames) + len(target_key)]
    for i in range(len(target_key)):
        if (key[i] != shift_key[i]):
            print(i, key[i], shift_key[i], bins[i])
