from theory import my_sin, fit_my_sin, fit_Q_Channel, angle_between_stokes_vectors
import numpy as np
import matplotlib.pyplot as plt


def measure_Coupling_Efficiency_cont(source, waveplates, timestamp):
    # Measure
    angle_max = 180
    deg_per_sec = 10

    source.turn_off()
    waveplates.home()
    source.turn_on(1)

    waveplates.jog_like_HWP(deg_per_sec)
    timestamp.read(int(angle_max / deg_per_sec))
    waveplates.stop()

    return timestamp.get_counts_per_second()


def evaluate_Coupling_Efficiency(times, counts):
    counts = np.array([[
        count[0] / sum(count), count[1] / sum(count), count[2] / sum(count),
        count[3] / sum(count)
    ] for count in counts])

    xs = np.linspace(times[0], times[-1], 10000)
    ps = []
    for i in range(4):

        plt.scatter(times, counts[:, i], label=i)

        p = fit_my_sin(times, counts[:, i])
        ps.append(p)
        plt.plot(xs,
                 my_sin(xs, *p),
                 linestyle="dashed",
                 color='dimgrey',
                 linewidth=2)
    plt.xlabel("Angle of HWP[°]")
    plt.ylabel("Count rates of channels normalised to overall count rate")
    plt.legend()

    plt.show()

    beamsplitterratio = (ps[0][0] + ps[1][0]) / (ps[2][0] + ps[3][0])

    print("The beamsplitter ratio of HV/PM is: ", beamsplitterratio)

    effH = ps[0][0] / (ps[0][0] + ps[1][0])  #+ps[2][0]+ps[3][0])
    effV = ps[1][0] / (ps[0][0] + ps[1][0])  #+ps[2][0]+ps[3][0])
    effP = ps[2][0] / (ps[2][0] + ps[3][0])  #+ps[2][0]+ps[3][0])
    effM = ps[3][0] / (ps[2][0] + ps[3][0])  #+ps[2][0]+ps[3][0])

    print("eff H = ", effH)
    print("eff V = ", effV)
    print("eff P = ", effP)
    print("eff M = ", effM)

    counts_corr_H = (counts[:, 0] - ps[0][3]) / effH
    counts_corr_V = (counts[:, 1] - ps[1][3]) / effV
    counts_corr_P = (counts[:, 2] - ps[2][3]) / effP
    counts_corr_M = (counts[:, 3] - ps[3][3]) / effM
    plt.plot(times, counts_corr_H)
    plt.plot(times, counts_corr_V)
    plt.plot(times, counts_corr_P)
    plt.plot(times, counts_corr_M)

    print('Phase of H in deg: ', ps[0][2] / (2 * np.pi) * 360)
    print('Phase of V in deg: ', ps[1][2] / (2 * np.pi) * 360)
    print('Phase of P in deg: ', ps[2][2] / (2 * np.pi) * 360)
    print('Phase of M in deg: ', ps[3][2] / (2 * np.pi) * 360)
    print('Freq of H: ', ps[0][1])
    print('Freq of V: ', ps[1][1])
    print('Freq of P: ', ps[2][1])
    print('Freq of M: ', ps[3][1])
    plt.axvline(ps[0][2] / (np.pi * 2) * 360, color='b')
    plt.axvline(ps[1][2] / (np.pi * 2) * 360, color='b')
    plt.axvline(ps[2][2] / (np.pi * 2) * 360, color='b')
    plt.axvline(ps[3][2] / (np.pi * 2) * 360, color='b')

    plt.title("Corrected Measurements")

    plt.show()

    plt.scatter(
        times, counts[:, 0] / effH + counts[:, 1] / effV +
        counts[:, 2] / effP + counts[:, 3] / effM)

    plt.show()

    return np.array([effH, effV, effP, effM])


def get_Coupling_Efficiency_cont(source, waveplates, timestamp):
    times, counts = measure_Coupling_Efficiency_cont(source, waveplates,
                                                    timestamp)
    return evaluate_Coupling_Efficiency(times, counts)


def measure_Polcomp_Counts(source, waveplates, timestamp):
    measurements = []
    bins = []
    channels = [1,3]
    measurementTime = 5
    waveplates.move_to(0)

    # Measure H using unitary
    source.turn_on(channels[0])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    bins.append(times)
    source.turn_off()

    # Measure P using unitary
    source.turn_on(channels[1])
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
    source.turn_on(channels[0])
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


def evaluate_Polcomp_Counts(counts, coupling):
    print("\nEVAL\n")
    print(counts)

    meas_V = np.concatenate((counts[0], counts[3, [2, 3]]))
    meas_M = np.concatenate((counts[1], counts[2, [2, 3]]))

    measurement_raw = np.vstack((meas_V, meas_M))

    print("Raw counts:\n", measurement_raw)

    measurement_dark = np.concatenate((counts[4], counts[4, 2:4]))

    print("Raw backround:\n", measurement_dark)
    measurement_darksubs = measurement_raw - measurement_dark

    print("Backround substracted measurement\n", measurement_darksubs)

    coupling = np.concatenate((coupling, coupling[2:4]))

    #for meas in measurement:
    #    meas_coup.append(np.hstack((meas[0:4] / eta, meas[4:6] / eta[2:4])))
    #
    #    meas_coup = np.array(meas_coup)

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

    v1 = [
        (matrix[0][0] - matrix[0][1])/(matrix[0][0]+matrix[0][1]), (matrix[0][3] - matrix[0][2])/(matrix[0][3]+matrix[0][2]),
        (matrix[0][4] - matrix[0][5])/(matrix[0][4]+matrix[0][5])
    ]
    v2 = [
        (matrix[1][0] - matrix[1][1])/(matrix[1][0]+matrix[1][1]), (matrix[1][3] - matrix[1][2])/(matrix[1][3]+matrix[1][2]),
        (matrix[1][4] - matrix[1][5])/(matrix[1][4]+matrix[1][5])
    ]

    print(v1,v2)
    a = angle_between_stokes_vectors(v1, v2)
    print(a)

    # Create a zero row with the same number of columns as your matrix
    zero_row = np.zeros(matrix.shape[1])

    # Insert zero_row as first row
    matrix = np.insert(matrix, 0, zero_row, axis=0)

    # Insert zero_row between second and third row
    matrix = np.insert(matrix, 2, zero_row, axis=0)

    # Alter the elements in rows 2 and 4
    matrix[0, [0, 1, 2, 3, 4, 5]] = matrix[1, [1, 0, 3, 2, 5, 4]]
    matrix[2, [0, 1, 2, 3, 4, 5]] = matrix[3, [1, 0, 3, 2, 5, 4]]

    print(matrix)

    angles, loss = fit_Q_Channel(matrix)

    return angles, loss, a
