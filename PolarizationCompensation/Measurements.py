from PolarizationCompensation.Theory import my_sin, fit_my_sin, fit_Q_Channel, angle_between_stokes_vectors
import numpy as np
import matplotlib.pyplot as plt


def measure_Coupling_Efficiency_cont(source, waveplates, timestamp):
    # Measure
    angle_max = 90
    deg_per_sec = 10

    source.turn_off()
    waveplates.home()
    source.turn_on("V")

    waveplates.jog_like_HWP(deg_per_sec)
    timestamp.read(int(angle_max / deg_per_sec))
    waveplates.stop()

    return timestamp.get_counts_per_second()


def evaluate_Coupling_Efficiency(counts,
                                 inv_symbol_map=["H", "V", "P", "M"],
                                 folder="./"):
    counts = np.array([[
        count[0] / sum(count), count[1] / sum(count), count[2] / sum(count),
        count[3] / sum(count)
    ] for count in counts])
    times = np.arange(0, len(counts), 1)
    xs = np.linspace(times[0], times[-1], 10000)
    ps = []
    plt.figure(figsize=(5, 4), dpi=400)
    for i in range(4):

        plt.scatter(times, counts[:, i], label=inv_symbol_map[i])

        p = fit_my_sin(times, counts[:, i])
        ps.append(p)
        plt.plot(xs,
                 my_sin(xs, *p),
                 linestyle="dashed",
                 color='dimgrey',
                 linewidth=2)
    plt.xlabel("Measurement time in s")
    plt.ylabel("Normalized count rates")
    plt.title("Sending linear polarization\nwhile rotating HWP 180deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder + "rotating_HWP.png")

    beamsplitterratio = (ps[0][0] + ps[1][0]) / (ps[2][0] + ps[3][0])

    print("The beamsplitter ratio of HV/PM is: ", beamsplitterratio)

    effH = ps[0][0] / (ps[0][0] + ps[1][0])
    effV = ps[1][0] / (ps[0][0] + ps[1][0])
    effP = ps[2][0] / (ps[2][0] + ps[3][0])
    effM = ps[3][0] / (ps[2][0] + ps[3][0])

    effs = [effH, effV, effP, effM]

    counts_corr = []
    plt.figure(figsize=(5, 4), dpi=400)
    for i in range(len(effs)):
        print("eff {} = {}".format(inv_symbol_map[i], effs[i]))
        counts_corr.append(counts[:, i] / effs[i])
        plt.plot(times, counts_corr[-1], label=inv_symbol_map[i])
    plt.xlabel("Measurement time in s")
    plt.ylabel("Corrected normalized count rates")
    plt.title("Sending linear polarization\nwhile rotating HWP 180deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder + "corrected_HWP.png")

    print('Phase of H in deg: ', ps[0][2] / (2 * np.pi) * 360)
    print('Phase of V in deg: ', ps[1][2] / (2 * np.pi) * 360)
    print('Phase of P in deg: ', ps[2][2] / (2 * np.pi) * 360)
    print('Phase of M in deg: ', ps[3][2] / (2 * np.pi) * 360)
    print('Freq of H: ', ps[0][1])
    print('Freq of V: ', ps[1][1])
    print('Freq of P: ', ps[2][1])
    print('Freq of M: ', ps[3][1])

    counts_sum = np.sum(counts_corr, axis=0)
    counts_sum /= max(counts_sum) / 100
    plt.figure(figsize=(5, 4), dpi=400)
    plt.scatter(times, counts_sum)
    plt.xlabel("Measurement time in s")
    plt.ylabel("Percent of max counts")
    plt.title("Sending linear polarization\nwhile rotating HWP 180deg")
    plt.tight_layout()
    plt.savefig(folder + "summed_HWP.png")

    return np.array([effH, effV, effP, effM])


def get_Coupling_Efficiency_cont(source, waveplates, timestamp):
    counts, _ = measure_Coupling_Efficiency_cont(source, waveplates, timestamp)
    return evaluate_Coupling_Efficiency(counts)


def measure_Polcomp_Counts(source, waveplates, timestamp, send_pol=["V", "M"]):
    measurements = []
    bins = []
    measurementTime = 5
    waveplates.move_to(0)

    # Measure pol0 using unitary
    source.turn_on(send_pol[0])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    bins.append(times)
    source.turn_off()

    # Measure pol1 using unitary
    source.turn_on(send_pol[1])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    bins.append(times)

    # Measure H using l4
    # Move "3 Waveplates" such that they represent a single lamba/4 at 0 deg
    waveplates.move_like_QWP(0)

    # Measure P using QWP at 0 deg
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    bins.append(times)
    source.turn_off()

    # Measure H using QWP at 0deg
    source.turn_on(send_pol[0])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    bins.append(times)
    source.turn_off()

    # Measure dark counts
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    bins.append(times)

    return bins, np.mean(measurements, axis=1)


def evaluate_Polcomp_Counts(counts,
                            coupling,
                            sent_pol=["V", "M"],
                            waveplate_setup=["QWP", "HWP", "QWP"]):
    print(counts)
    meas_1 = np.concatenate((counts[0], counts[3, [2, 3]]))
    meas_2 = np.concatenate((counts[1], counts[2, [2, 3]]))
    measurement_raw = np.vstack((meas_1, meas_2))
    print("Raw counts:\n", measurement_raw)

    measurement_dark = np.concatenate((counts[4], counts[4, 2:4]))
    print("Raw backround:\n", measurement_dark)

    measurement_darksubs = measurement_raw - measurement_dark
    print("Backround substracted measurement\n", measurement_darksubs)

    coupling = np.concatenate((coupling, coupling[2:4]))
    measurement_eff = measurement_darksubs / coupling
    print("Backround substracted and eff corrr measurement\n", measurement_eff)

    measurement_statistics = []
    for i in range(len(measurement_eff)):
        measurement_statistics.append(
            np.array([
                measurement_eff[i, 0:2] / np.sum(measurement_eff[i, 0:2]),
                measurement_eff[i, 2:4] / np.sum(measurement_eff[i, 2:4]),
                measurement_eff[i, 4:6] / np.sum(measurement_eff[i, 4:6])
            ]).flatten())
    measurement_statistics = np.array(measurement_statistics)
    print("Normalized Measurements: \n{}".format(measurement_statistics))

    # Expand vom VM
    matrix = measurement_statistics

    v1 = [(matrix[0][0] - matrix[0][1]) / (matrix[0][0] + matrix[0][1]),
          (matrix[0][3] - matrix[0][2]) / (matrix[0][3] + matrix[0][2]),
          (matrix[0][4] - matrix[0][5]) / (matrix[0][4] + matrix[0][5])]
    v2 = [(matrix[1][0] - matrix[1][1]) / (matrix[1][0] + matrix[1][1]),
          (matrix[1][3] - matrix[1][2]) / (matrix[1][3] + matrix[1][2]),
          (matrix[1][4] - matrix[1][5]) / (matrix[1][4] + matrix[1][5])]

    a = angle_between_stokes_vectors(v1, v2)
    print("Angle between the stokes vectors: {}".format(a))

    # Create a zero row with the same number of columns as your matrix
    zero_row = np.zeros(matrix.shape[1])

    for pol in sent_pol:
        if pol == "H":
            # Insert zero_row as second (V) row
            matrix = np.insert(matrix, 1, zero_row, axis=0)
            # Add values inverse to H
            matrix[1, [0, 1, 2, 3, 4, 5]] = matrix[0, [1, 0, 3, 2, 5, 4]]
        if pol == "V":
            matrix = np.insert(matrix, 0, zero_row, axis=0)
            matrix[0, [0, 1, 2, 3, 4, 5]] = matrix[1, [1, 0, 3, 2, 5, 4]]
        if pol == "P":
            matrix = np.insert(matrix, 3, zero_row, axis=0)
            matrix[3, [0, 1, 2, 3, 4, 5]] = matrix[2, [1, 0, 3, 2, 5, 4]]
        if pol == "M":
            matrix = np.insert(matrix, 2, zero_row, axis=0)
            matrix[2, [0, 1, 2, 3, 4, 5]] = matrix[3, [1, 0, 3, 2, 5, 4]]

    angles, loss = fit_Q_Channel(matrix, waveplate_setup)

    return angles, loss, a
