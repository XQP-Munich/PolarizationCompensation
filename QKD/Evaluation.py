import numpy as np
from scipy import signal as sgnl
from scipy import stats as stats
import time

dt = 10000
filter = 1.32 / 100
filter = 0.0001  #40dBloss
bob_eff = filter * 25.3852 / 100
alice_trans = 0.917 / 0.9954
rep_rate = 100E6
chan_offset = [12000, 12071, 10000 - 4515, 13070, 12000]  # todo remove global


def guess_key(bins):
    key = []
    qbers = []
    num_sifted_det = []
    for count in bins:
        diff1 = count[0] - count[1]
        diff2 = count[2] - count[3]
        diff3 = abs(diff1) - abs(diff2)
        val = 4
        qber = 0.5
        if diff3 > 0:
            basis_counts = count[0] + count[1]
            if basis_counts > 0:
                if diff1 > 0:
                    val = 0
                    qber = count[1] / basis_counts
                else:
                    val = 1
                    qber = count[0] / basis_counts
        else:
            basis_counts = count[2] + count[3]
            if basis_counts > 0:
                if diff2 > 0:
                    val = 2
                    qber = count[3] / basis_counts
                else:
                    val = 3
                    qber = count[2] / basis_counts
        num_sifted_det.append(basis_counts)
        key.append(val)
        qbers.append(qber)

    return np.array(key), np.array(qbers), np.array(num_sifted_det)


def find_filter(phases, height=0.8):
    bin_width = 1000
    filter_min = []
    filter_max = []
    shifts = []
    target_shifts = [0.4, 0.4, 0.4, 0.4, 0.1]
    heights = [height] * 4 + [0.5]
    for i in range(5):
        bins = np.histogram(phases[1, phases[0] == i + 1],
                            bins=bin_width,
                            range=(0, dt))[0]
        peaks, _ = sgnl.find_peaks(bins, prominence=max(bins) / 2)
        shift = int(bin_width * target_shifts[i] - peaks[0])
        shifts.append(int(shift * dt / bin_width))
        bins = np.roll(bins, shift)
        peaks += shift
        _, _, f_min, f_max = sgnl.peak_widths(bins,
                                              peaks,
                                              rel_height=heights[i],
                                              wlen=bin_width)
        filter_min.append(f_min[0] * dt / bin_width)
        filter_max.append(f_max[0] * dt / bin_width)
    return np.array(shifts), np.array([
        np.mean(filter_min[:4]),
        np.mean(filter_max[:4]), filter_min[4], filter_max[4]
    ])


def evaluate_tags(data,
                  channels,
                  offsets,
                  sent_key=None,
                  key_length=None,
                  sync_offset=np.nan,
                  filters=None,
                  verbose=True,
                  time_filtering=True):
    sync_data = data[1][data[0] == channels["SYNC"]["ch"]]
    sync_phase = 0
    #sync_offset = 6
    if sent_key is not None:
        key_length = len(sent_key)
    if sync_data.size != 0:
        key_length_sync = int(np.rint(np.median(np.diff(sync_data)) / dt))
        if key_length is not None and key_length_sync != key_length:
            print("Target key length {} does not match detected key length {}".
                  format(key_length, key_length_sync))
            return []
        key_length = key_length_sync
        if verbose:
            print("Sync Chan detected")
            print("Key length: {}".format(key_length))
        t0 = sync_data[1]
        print(t0 * 1E-12)
        data[1] -= t0 - 1000
        data = data[:,
                    np.logical_and(data[1] >= 0, data[1] <= (sync_data[-1] -
                                                             sync_data[1]))]
        data[1] += t0
        sync_data = data[1][data[0] == channels["SYNC"]["ch"]]
        sync_phase = int(np.rint(stats.circmean(sync_data % dt, high=dt)))
        if verbose:
            print("Phase: {}".format(sync_phase))
        meas_time = (data[1][-1] - data[1][0]) * 1E-12
        del sync_data
    else:
        t0 = data[1][1]
        meas_time = (data[1][-1] - t0) * 1E-12
        data[1] -= t0
        data = data[:, data[1] >= 0]
        sync_phase = 0
        if key_length is None:
            key_length = 1

    if verbose:
        print("Got {:.2e} Events in {:.0f}s".format(len(data[0]), meas_time))
        print("Got {:.2e} Photon Events in {:.0f}s".format(
            len(data[0, data[0] != channels["SYNC"]["ch"]]), meas_time))

    if filters is None:
        temp_data = data.copy()
        for i in range(0, 5):
            temp_data[1, temp_data[0] == i + 1] += offsets[i]
        try:
            shifts, filters, = find_filter(temp_data % dt)
            offsets += shifts + sync_phase
        except:
            print("Unable to find peaks")
            offsets = np.array(offsets) + sync_phase
            filters = np.array([0, dt, 0, dt])

    if not time_filtering:
        filters = np.array([0, dt, 0, dt])

    if verbose:
        print("Offsets: {}".format(offsets))

    for i in range(0, 5):
        data[1, data[0] == i + 1] += offsets[i] - sync_phase

    del sync_phase

    phases = np.vstack(
        (data % dt, ((data[1] % (dt * key_length)) / dt).astype(int)))

    bg_time_mask = np.logical_and(
        np.logical_and(np.logical_and(phases[1] >= 0, phases[1] <= 2000),
                       np.logical_and(phases[1] >= 8000, phases[1] <= 10000)),
        data[0] != channels["SYNC"]["ch"])

    data_bg = data[:, bg_time_mask]
    bg_cps = len(data_bg[0]) / meas_time / 4 * 10
    print("BG CPS: {}".format(bg_cps))

    time_mask = np.logical_or(
        np.logical_and(phases[1] >= filters[0], phases[1] <= filters[1]),
        data[0] == channels["SYNC"]["ch"])
    datan = data[:, ~time_mask]
    data = data[:, time_mask]
    print(len(data[1, data[0] != 5]), len(datan[1, datan[0] != 5]))
    #print("S/N: {}".format(
    #    len(data[1, data[0] != channels["SYNC"]["ch"]]) /
    #    len(datan[1, datan[0] != channels["SYNC"]["ch"]])))

    sifted_events = data[:, data[0] != 5]
    sifted_events[1] = (sifted_events[1]) // dt

    bins = np.zeros((key_length, 4))
    pos_in_key = ((data[1] % (dt * key_length)) / dt).astype(int)
    for i in range(0, 4):
        pos_in_key_pol = pos_in_key[data[0] == i + 1]
        unique_indices, counts = np.unique(pos_in_key_pol, return_counts=True)
        for id, cnt in zip(unique_indices, counts):
            bins[id][i] += cnt

    key_guess, qbers, num_sifted_det = guess_key(bins)
    keymatch = None
    if sent_key is not None and np.isnan(sync_offset):
        matching_symbols, sync_offset = compare_key(sent_key, key_guess)
        if True:
            if (matching_symbols == key_length):
                print("Keys match with offset: {}".format(sync_offset))
            else:
                keymatch = "{}/{}".format(matching_symbols, key_length)

                print("Keys match with {}/{} symbols, most likely offset: {}".
                      format(matching_symbols, key_length, sync_offset))
        if matching_symbols <= key_length * 0.25:
            sync_offset = np.nan

    used_key = key_guess
    if sent_key is not None and not np.isnan(sync_offset):
        print("using sent_key")
        used_key = sent_key
        used_key = np.array(
            (used_key + used_key)[sync_offset:sync_offset + key_length])

    mcount = [[], [], [], [], [], [], [], []]
    if False:
        for i, b in enumerate(bins):
            if used_key[i] != key_guess[i]:
                print("*", end="")
            if b.sum() <= np.mean(np.sum(bins, axis=1)) / 2:
                print("#", end="")
            if b.sum() >= np.mean(np.sum(bins, axis=1)) * 2:
                print("%", end="")
            print(inv_state_map[used_key[i]], inv_state_map[key_guess[i]], b,
                  "{:.2f}".format(qbers[i] * 100), num_sifted_det[i])
            mcount[used_key[i]].append(b.sum())
        for c in mcount:
            print(np.mean(c), np.std(c))
        for c in mcount:
            print(c)
    if sent_key is not None and not np.isnan(sync_offset):
        phases[2] = used_key[phases[2]]
    else:
        phases[2] = phases[0]

    if verbose:
        print("Mean Qber: {:.2f}% with std of {:.4f}".format(
            np.nanmean(qbers) * 100,
            np.nanstd(qbers) * 100))
        print("Max Qber: {:.2f}% at {}".format(
            np.max(qbers) * 100, bins[np.argmax(qbers)]))
        print("Sifted key Rate: {:.2f}".format(
            np.sum(num_sifted_det) / meas_time))

    pol_clicks = []
    for pol in range(8):
        pol_bin = bins[used_key == pol, :]
        pol_clicks.append(pol_bin.sum(axis=1))
    return [
        phases, offsets, filters, sync_offset, qbers, num_sifted_det,
        meas_time, t0 * 1E-12, sifted_events, pol_clicks, keymatch, bg_cps
    ]


def process_data(data,
                 frame,
                 channels,
                 offsets=chan_offset,
                 sync_offset=np.nan,
                 sent_key=None,
                 verbose=True,
                 time_filtering=True,
                 last_valid_sync=0,
                 tm=0):
    print("Evaluating Frame {}".format(frame))
    start = time.time()

    if len(data[0]) <= 0:
        print("No Timestamps in frame")
        return frame, None
    valid_data, last_valid_sync = get_valid_frames(
        data, channels, last_valid_sync=last_valid_sync, verbose=False)
    if len(valid_data) <= 0:
        print("No Sync detected in frame")
        if np.isnan(sync_offset):
            valid_data = data
        else:
            print("Skipping Frame")
            return frame, None
    et = evaluate_tags(valid_data,
                       channels,
                       offsets,
                       sync_offset=sync_offset,
                       sent_key=sent_key,
                       verbose=verbose,
                       time_filtering=time_filtering)
    if len(et) == 0:
        print("Evaluation error, continue")
        return frame, None
    phases, offsets, filters, sync_offset = et[:4]
    qbers, num_sifted_det, meas_time, t0 = et[4:8]
    sifted_events, pol_clicks, key_match, bg_cps = et[8:]

    #self.verbose = False
    nmax = []
    hists = []
    bins = []
    phases_pol = phases[:, phases[0] != 5]
    for pol in range(8):
        hist, bins = np.histogram(phases_pol[1, phases_pol[2] == pol],
                                  bins=100,
                                  range=(0, dt))
        hists.append(hist / 1000)
        nmax.append(np.max(hist) / 1000)
    hist, bins = np.histogram(phases[1, phases[0] == 5],
                              bins=100,
                              range=(0, dt))
    hists.append(hist)
    nmax.append(np.max(hist))

    hists[-1] = hists[-1] * (np.max(nmax[:-1]) / nmax[-1])
    phase_data = [hists, bins, filters]
    cps_data = [frame, pol_clicks, meas_time, tm, bg_cps]
    ui_data = [
        frame,
        qbers,
        np.sum(num_sifted_det) / meas_time / 1000,
        sync_offset,
        offsets,
        key_match,
    ]
    eval_data = [offsets, sync_offset, last_valid_sync]
    duration = time.time() - start

    print("eval took {:.2f}s".format(duration))
    return frame, [eval_data, phase_data, cps_data, ui_data, sifted_events]


def get_valid_frames(data, channels, last_valid_sync=0, verbose=True):
    valid_timing = 500  # in ps
    sync_indices = np.where(data[0] == channels["SYNC"]["ch"])[0]
    if len(sync_indices) <= 20:
        return []
    sync_diffs = np.diff(data[1][sync_indices], append=0)
    sync_diffs_median = np.median(sync_diffs)
    #sync_diffs_median = 4096*10E-9
    sync_dropouts = np.where(
        np.logical_or(sync_diffs > (sync_diffs_median + valid_timing),
                      sync_diffs < (sync_diffs_median - valid_timing)))[0]
    if len(sync_dropouts) > 1:
        print("{} SYNC dropouts detected:\n{}".format(
            len(sync_dropouts), sync_indices[sync_dropouts]))
    sync_dropouts = np.concatenate(([-1], sync_dropouts))
    valid_frames = []
    for i in range(len(sync_dropouts) - 1):
        beg = sync_indices[sync_dropouts[i] + 1]
        end = sync_indices[sync_dropouts[i + 1]]
        if end > beg:
            d = data[:, beg:end + 1]
            #print(d)
            d[1] -= d[1, 0] - last_valid_sync - np.int64(sync_diffs_median)
            #print(d)
            last_valid_sync = d[1, -1]
            valid_frames.append(d)
    return np.concatenate(valid_frames, axis=1), last_valid_sync


def compare_key(sent_key, detected_key):
    sames = []
    for i in range(len(sent_key)):
        sames.append(
            np.count_nonzero(detected_key == np.roll(sent_key, -i) % 4))
    return max(sames), np.argmax(sames)
