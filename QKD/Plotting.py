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


def plot_mus(canvas,
             plot_data,
             loss,
             rep_rate=100E6,
             alice_loss=0,
             bob_loss=0):
    frame, clicks, meas_time, tm, _ = plot_data
    key_length = sum(len(x) for x in clicks)
    mus = []
    for i in range(len(clicks)):
        mus.append(clicks[i] / meas_time / rep_rate / (1 / key_length) /
                   (10**(bob_loss / 10)) / (10**(loss / 10)) *
                   (10**(alice_loss / 10)))

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


def plot_floss(canvas,
               plot_data,
               mu,
               rep_rate=100E6,
               alice_loss=0,
               bob_loss=0):
    frame, clicks, meas_time, tm, bg_cps = plot_data
    expected_cps = rep_rate * mu * (10**(bob_loss / 10))

    key_length = sum(len(x) for x in clicks)
    chan_clicks = []
    for i in range(4):
        chan_clicks.append(np.mean(np.concatenate((clicks[i], clicks[i + 4]))))
    chan_cps = np.array(chan_clicks) / meas_time * key_length

    loss = np.log10(chan_cps / expected_cps) * 10

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
    canvas.ydata.append(loss[canvas.pol])
    ax.plot(canvas.xdata, canvas.ydata, label="Loss")
    ax.legend(loc="center left")

    #ax.set_title("μ={:.2f}±{:.2f}  ν={:.2f}±{:.2f}".format(

    ax.set_xlabel("Frame")
    ax.set_ylabel("Loss in dB")
    canvas.fig.tight_layout()
    print("loss plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


def plot_qber(canvas, ui_data):
    frame, qber, _, _, _, _ = ui_data
    qber = np.mean(qber) * 100

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
    canvas.ydata.append(qber)
    ax.plot(canvas.xdata,
            canvas.ydata,
            label="qber",
            color="C{}".format(canvas.pol + 4))
    ax.legend(loc="center left")

    ax.set_title("QBER {:.2f}%".format(qber))

    ax.set_xlabel("Frame")
    ax.set_ylabel("QBER in %")
    canvas.fig.tight_layout()
    print("qber plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


def plot_skr(canvas, ui_data):
    frame, _, skr, _, _, _ = ui_data

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
    canvas.ydata.append(skr)
    ax.plot(canvas.xdata,
            canvas.ydata,
            label="skr",
            color="C{}".format(canvas.pol + 4))
    ax.legend(loc="center left")

    ax.set_title("Sifted Key Rate {:.2f}Kbit/s".format(skr))

    ax.set_xlabel("Frame")
    ax.set_ylabel("Sifted Key Rate in Kbit/s")
    canvas.fig.tight_layout()
    print("skr plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


def plot_lloss(canvas,
               plot_data,
               lmu,
               rep_rate=100E6,
               alice_loss=0,
               bob_loss=0):
    frame, clicks, meas_time, tm, bg_cps = plot_data
    lmu = np.array(lmu).transpose()
    if len(lmu) <= 0:
        print("No Alice Data")
        return
    interv = (tm - meas_time, tm)
    for t in interv:
        if t < lmu[0][0] - 10 or t > lmu[0][-1] + 10:
            print("{} is not in [{},{}]".format(t, lmu[0][0], lmu[0][-1]))
            return
    xs = np.arange(*interv, 1)
    lmut = np.interp(xs, lmu[0], lmu[1])
    mu = np.mean(lmut)

    expected_cps = rep_rate * mu * (10**(bob_loss / 10))  #*bob_optics

    pols_sent = [len(x) for x in clicks]

    chan_clicks = []
    for i in range(4):
        chan_clicks.append(
            np.sum(np.concatenate((clicks[i], clicks[i + 4]))) /
            (pols_sent[i] + pols_sent[i + 4]))
    chan_cps = np.array(chan_clicks) / meas_time * np.sum(pols_sent) - bg_cps

    print(chan_cps, expected_cps)

    loss = np.log10(chan_cps / expected_cps) * 10

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
    canvas.ydata.append(loss)
    ydat = np.transpose(canvas.ydata)
    for i, pol in enumerate(["H", "V", "P", "M"]):
        ax.plot(canvas.xdata, ydat[i], label=pol, color="C{}".format(i))
    ax.legend(loc="center left")

    ax.set_title("{:.2f}dB Channel loss".format(np.mean(loss)))

    ax.set_xlabel("Frame")
    ax.set_ylabel("Loss in dB")
    canvas.fig.tight_layout()
    print("loss plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True


def plot_lmu(canvas, plot_data, lmu, rep_rate=100E6, alice_loss=0, bob_loss=0):
    frame, cps, meas_time, tm, _ = plot_data
    lmu = np.array(lmu).transpose()
    if len(lmu) <= 0:
        print("No Alice Data")
        return
    interv = (tm - meas_time, tm)
    for t in interv:
        if t < lmu[0][0] - 10 or t > lmu[0][-1] + 10:
            print("{} is not in [{},{}]".format(t, lmu[0][0], lmu[0][-1]))
            return
    xs = np.arange(*interv, 1)
    lmut = np.interp(xs, lmu[0], lmu[1])
    mu = np.mean(lmut)

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
    canvas.ydata.append(mu)
    ax.plot(canvas.xdata,
            canvas.ydata,
            label="μ",
            color="C{}".format(canvas.pol + 4))
    ax.legend(loc="center left")

    ax.set_title("Alice brightness μ={:.2f}".format(mu))

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean photon Number")
    canvas.fig.tight_layout()
    print("lmu plot took {:.2f}s".format(time.time() - start))
    canvas.draw()
    return True
