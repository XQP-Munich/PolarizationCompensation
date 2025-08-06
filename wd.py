
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest

settings = Settings(  # default
  ts_shape="long",  # tidy data
  ts_humanize=True,  # humanized parameters
  ts_convert_units=True  # convert values to SI units
)

request = DwdObservationRequest(
  parameters=[
    ("hourly", "cloudiness"),
    ],
  start_date="2024-09-11",  # if not given timezone defaulted to UTC
  end_date="2024-09-13",  # if not given timezone defaulted to UTC
  settings=settings
).filter_by_station_id(station_id=(7431,3379))

stations = request.df
stations.head()
print(stations)
values = request.values.all().df  
values.head()
print(values.to_numpy())
