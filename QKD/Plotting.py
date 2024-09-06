import time
import numpy as np

state_map = {"H": 0, "V": 1, "P": 2, "M": 3, "h": 4, "v": 5, "p": 6, "m": 7}
inv_state_map = {v: k for k, v in state_map.items()}


def plot_phases(canvas, plot_data):
    hists, bins, filters = plot_data
    ax = canvas.axes
    ax.cla()
    start = time.time()
    hist_sig = hists[canvas.pol]
    hist_dec = hists[canvas.pol + 4]
    hist_other = np.sum([
        hist for i, hist in enumerate(hists[:-1])
        if (i != canvas.pol and i != canvas.pol + 4)
    ],
                        axis=0) / 3
    ax.bar(bins[:-1],
           hist_sig,
           width=100,
           alpha=0.4,
           label=inv_state_map[canvas.pol] + " Signal",
           color="C{}".format(canvas.pol))
    ax.bar(bins[:-1],
           hist_dec,
           width=100,
           alpha=0.4,
           label=inv_state_map[canvas.pol] + " Decoy",
           color="C{}".format(canvas.pol + 4))
    ax.bar(bins[:-1],
           hist_other,
           width=100,
           alpha=0.2,
           label="Rest",
           color="gray")
    ax.bar(bins[:-1],
           hists[-1],
           width=100,
           alpha=0.4,
           label="SYNC*",
           color="C{}".format(5))
    ax.vlines(filters[0], 0, max(hists[-1]))
    ax.vlines(filters[1], 0, max(hists[-1]))
    ax.annotate("Δ  = {:.0f}ps".format((filters[1] - filters[0])),
                (filters[0] - 100, max(hists[-1]) * 0.9),
                fontsize="8",
                ha="right",
                va="top")
    ax.annotate("FWHM = {:.0f}ps".format((filters[3] - filters[2])),
                (filters[3] + 100, 20),
                fontsize="8")
    ax.legend()
    ax.set_xlabel("Detection time in ps")
    ax.set_ylabel("Number of detections")
    canvas.fig.tight_layout()
    print("phase plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


def plot_mus(canvas, plot_data):
    frame, mus, = plot_data
    ax = canvas.axes
    ax.cla()
    start = time.time()
    if canvas.xdata is None:
        canvas.xdata = []
        canvas.ydata = []
    if len(canvas.xdata) >= 30:
        canvas.xdata = canvas.xdata[1:]
        canvas.ydata = canvas.ydata[1:]
    canvas.xdata.append(frame)
    canvas.ydata.append([
        np.mean(mus[canvas.pol]),
        np.mean(
            [np.mean(mu) for i, mu in enumerate(mus[:4]) if i != canvas.pol]),
        np.mean(mus[canvas.pol + 4]),
        np.mean(
            [np.mean(mu) for i, mu in enumerate(mus[4:]) if i != canvas.pol])
    ])
    ax.plot(canvas.xdata, canvas.ydata, label=["μ", "μ_r", "ν", "ν_r"])
    ax.legend(loc="center left")
    ax.hlines([0.51, 0.15],
              canvas.xdata[0],
              canvas.xdata[-1],
              linestyles="dashed",
              colors="gray")

    ax.set_title("μ={:.2f}±{:.2f}  ν={:.2f}±{:.2f}".format(
        np.mean(mus[canvas.pol]), np.std(mus[canvas.pol]),
        np.mean(mus[canvas.pol + 4]), np.std(mus[canvas.pol + 4])))

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean photon number")
    canvas.fig.tight_layout()
    print("mu plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True
