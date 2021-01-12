from tdmtpy import Configure
from tdmtpy.utils import RGF_from_SW4
from obspy.core import UTCDateTime

# Inputs
config = Configure(path_to_file="_tmp/sw4test/RGF2/mtinv.in")
path_to_green = "."
t0 = 0.2 # time shift from origin
file_name = "source1" # sw4 output file name (e.g. event name)

# Event and station information
origin_time = UTCDateTime(config.event["datetime"])
evla = config.event["latitude"]
evlo = config.event["longitude"]
depth = 2.0 # in km
stations = config.df.station.values
station_lat = config.df.latitude.values
station_lon = config.df.longitude.values

RGF_from_SW4(path_to_green=path_to_green, t0=t0, file_name=file_name,
             origin_time=origin_time,event_lat=evla,event_lon=evlo,depth=depth,
             station_name=stations,station_lat=station_lat,station_lon=station_lon)
