import Devices as dev
import time, os
from theory import my_sin, fit_my_sin
import numpy as np
import matplotlib.pyplot as plt


def get_Coupling_Efficiency_cont(folder, source, waveplates, timestamp):
    # Measure
    angle_max = 180
    deg_per_sec = 10

    source.turn_off()
    waveplates.home()
    source.turn_on("V")

    measurement = folder + "coupling_efficiency_{}/".format(
        time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(measurement)

    waveplates.jog_like_HWP(deg_per_sec)
    data = timestamp.read(int(angle_max / deg_per_sec))
    waveplates.stop()

    print(data)
    # Evaluate
    times, counts = timestamp.get_counts_per_second()
    #times = np.arange(0, angle_max / deg_per_sec + 1)

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

    np.savetxt(folder + "effs.txt", np.array([effH, effV, effP, effM]))

    return np.array([effH, effV, effP, effM])


def get_Polcomp_Angles(folder, source, waveplates, timestamp):
    measurements = []
    channels = [0, 3]
    measurementTime = 5
    waveplates.move_to(0)

    # Measure H using unitary
    source.turn_on(channels[0])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    source.turn_off()

    # Measure P using unitary
    source.turn_on(channels[1])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    source.turn_off()

    # Measure H using l4
    # Move "3 Waveplates" such that they represent a single lamba/4 at 0 deg
    waveplates.move_to_l4()

    source.turn_on(channels[0])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    source.turn_off()

    # Measure P using l4
    source.turn_on(channels[1])
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    source.turn_off()

    # Measure dark counts
    timestamp.read(measurementTime)
    times, counts = timestamp.get_counts_per_second()
    measurements.append(counts)
    return measurements
