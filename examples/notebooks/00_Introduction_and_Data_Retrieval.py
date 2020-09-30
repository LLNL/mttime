# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: obspy
#     language: python
#     name: obspy
# ---

# ### Example: 2019-07-16 Earthquake near Byron, California
# Prepared by Andrea Chiang, andrea@llnl.gov
#
# USGS event information URL https://earthquake.usgs.gov/earthquakes/eventpage/nc73225421/executive
#
# In this tutorial we will:
# * Download and process data.
# * Calculate Green's functions.
# * Calculate moment tensor using tdmtpy.
#
# Green's functions are computed using the software package Computer Porgrams in Seismology by Robert Herrmann (http://www.eas.slu.edu/eqc/eqccps.html).
#
# To run this tutorial you will need Python 3+ and the following packages:
# * ObsPy
# * pandas
# * matplotlib
# * NumPy
# * tdmtpy

# Import third-party libraries
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import read_events, UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader


# I have set the search variables to download only the earthquake of interest, and the quakeml file already exists.
#
# To download the event information change
# ```python
# event_bool = True
# ```

# +
event_bool = True

if event_bool:
    dataCenter="IRIS"
    client = Client(dataCenter)
    starttime = UTCDateTime("2019-07-16T00:00:00")
    endtime = UTCDateTime("2019-07-16T23:59:59")
    catalog = client.get_events(starttime=starttime, endtime=endtime,
                        minmagnitude=4,maxmagnitude=5,
                        minlatitude=36, maxlatitude=38,
                        minlongitude=-122, maxlongitude=-120)
    catalog.write("quakes.xml",format="QUAKEML")

# -

# ### Download data
# We will download the waveforms and station metadata from the Northern California Earthquake Data Center (NCEDC) using ObsPy's mass_downloader function.
#
# The next cell will create a directory for each event and all files will be stored there. In addition to MSEED and STATIONXML files we will also write the event origin information to a text file. This text file will be stored in the current working directory.

# +
dataCenter="NCEDC" 

# Time before and after event origin for waveform segments
time_before = 60
time_after = 300
download_bool = True

catalog = read_events("quakes.xml")
#catalog.plot(method="cartopy",projection="ortho")
for event in catalog:
    evid = str(catalog[0].origins[0].resource_id).split("=")[-1] # User origin resource id as the event id
    outdir = evid
    Path(outdir).mkdir(parents=True,exist_ok=True)
    
    # Event origin
    origin_time = event.preferred_origin().time
    starttime = origin_time - time_before
    endtime = origin_time + time_after
    
    # Event location
    evlo = event.preferred_origin().longitude
    evla = event.preferred_origin().latitude
    depth = event.preferred_origin().depth # in meters
    
    # Set the search area
    domain = CircularDomain(latitude=evla, longitude=evlo, minradius=0.7, maxradius=1.3)
    
    # Set the search period and additional criteria
    restrictions = Restrictions(starttime=starttime, endtime=endtime,
        reject_channels_with_gaps=True,
        minimum_length=0.95,
        network="BK",
        channel_priorities=["BH[ZNE12]", "HH[ZNE12]"],
        sanitize=True)
    
    # Save catalog info to file
    event_out = (
        "{evid:s},{origin:s},{jdate:s},"
        "{lon:.4f},{lat:.4f},{depth:.4f},"
        "{mag:.2f},{auth:s}\n"
        )        

    if event.preferred_magnitude() is None:
        mag = -999.
        magtype = "ml"
    else:
        mag = event.preferred_magnitude().mag
        magtype = event.preferred_magnitude().magnitude_type.lower()
    if event.preferred_origin().extra.catalog.value is None:
        auth = "unknown"
    else:
        auth = event.preferred_origin().extra.catalog.value.replace(" ","")
        
    event_out = event_out.format(
        evid=evid,
        origin=str(origin_time),
        jdate="%s%s"%(origin_time.year,origin_time.julday),
        lon=evlo,
        lat=evla,
        depth=depth/1E3,
        mag=mag,
        auth=auth
        )
        
    outfile = "datetime.csv"
    with open(outfile,"w") as f:
        f.write("evid,origin,jdate,lon,lat,depth,%s,auth\n"%magtype)
        f.write(event_out)
        
    # Dowanload waveforms and metadata
    if download_bool:
        mseed_storage = "%s/waveforms"%outdir
        stationxml_storage = "%s/stations"%outdir
        mdl = MassDownloader(providers=[dataCenter])
        mdl_helper = mdl.download(domain, restrictions,
            mseed_storage=mseed_storage,stationxml_storage=stationxml_storage)
        print("%s download completed"%outdir)
        
        
    print("%s is DONE."%outdir)


# -
# **Now we've downloaded the raw data, the next step is to process them.**

