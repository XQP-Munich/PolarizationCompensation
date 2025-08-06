from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest

from datetime import datetime, timedelta
import numpy as np

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

print(get_weather(datetime(2024,9,4,12),datetime(2024,9,4,18)))
