import Devices as dev
import time, os
from theory import my_sin, fit_my_sin
import numpy as np
import matplotlib.pyplot as plt


def get_Coupling_Efficiency_cont(folder, source, waveplates, timestamp):
    # Measure
    angle_max = 180
    deg_per_sec = 5

    source.turn_off()
    waveplates.move_to(0)
    source.turn_on("V")

    measurement = folder + "coupling_efficiency_{}/".format(
        time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(measurement)

    waveplates.jog_like_HWP(deg_per_sec)
    timestamp.read(measurement + "hwp.dat")
    time.sleep(int(angle_max / deg_per_sec) / 100)
    timestamp.stop()
    waveplates.stop()

    # Evaluate
    counts = timestamp.get_counts_per_second()
    times = np.arange(0, angle_max / deg_per_sec + 1)

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
