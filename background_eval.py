import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import sys

filenames = ["./data/background/background_with_3nm_07nm_2days.txt"]


from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest

settings = Settings(  # default
  ts_shape="long",  # tidy data
  ts_humanize=True,  # humanized parameters
  ts_convert_units=True  # convert values to SI units
)

def get_weather(start,end):
    request = DwdObservationRequest(
    parameters=[
        ("hourly", "cloudiness"),("hourly","precipitation")
        ],
    start_date=start-timedelta(hours=1),  # if not given timezone defaulted to UTC
    end_date=end-timedelta(hours=1),  # if not given timezone defaulted to UTC
    settings=settings
    ).filter_by_station_id(station_id=(7431,3379))

    stations = request.df
    stations.head()
    print(stations)
    values = request.values.all().df  
    values.head()
    print(values)
    values= values.to_numpy()
    values[:,3]+=timedelta(hours=1)
    clouds = values[values[:,2]=="cloud_cover_total"]
    clouds = clouds[:,3:5]
    clouds = clouds[clouds[:,1]>=0]
    print(clouds)
    prec = values[np.logical_and(values[:,2]=="precipitation_index",values[:,0]=="07431")]
    prec = prec[:,3:5]
    print(prec)
    return clouds, prec 


for filename in filenames:
    data = np.loadtxt(filename)[10:]
    data = data.transpose()


    t = np.array(data[0], dtype='datetime64[s]') + np.timedelta64(2, 'h')

    th = np.arange(t[0], t[-1], np.timedelta64(1, "h"))
    weather = get_weather(th[0].astype(datetime),th[-1].astype(datetime))

    pols = ["H", "V", "P", "M"]
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=600)
    ax2 = ax1.twinx()
    N = 60
    t = t[N - 1:]
    # for i in range(4):
    #     ax1.plot(
    #         t, data[i + 1][N - 1:], lw=0.1, label="{}".format(pols[i])
    #     )  # mean: {:.2f} std: {:.2f}".format(pols[i],np.mean(data[i+1]),np.std(data[i+1])))
    rolling_average = np.convolve(np.sum(data[1:], axis=0),
                                  np.ones(N) / N,
                                  mode="valid")
    ax1.plot(
        t, rolling_average, label="{}".format("Sum")
    )  # mean: {:.2f} std: {:.2f}".format("Sum",np.mean(np.sum(data[1:],axis=0)),np.std(np.sum(data[1:],axis=0))))
    ax2.plot(weather[0][:,0],weather[0][:,1],label="Cloudiness",c="C1")
    #ax2.plot(weather[1][:,0],weather[1][:,1],label="Prec",lw=0.5)


    #ax1.set_title(filename)
    ax1.set_title("Background Counts")
    ax1.set_ylabel("Counts / s")
    #ax1.set_xlabel("Time")
    ax2.set_ylabel("Cloudiness")
    #plt.gca().tick_params(axis='x', labelrotation=45)
    ax1.set_yscale("log")
    fig.autofmt_xdate(rotation=45)
    #ax1.legend(prop={'size': 4})
    #ax2.legend(prop={'size': 4})
    fig.tight_layout()
    fig.savefig(filename[:-3] + "png")
    #plt.show()
