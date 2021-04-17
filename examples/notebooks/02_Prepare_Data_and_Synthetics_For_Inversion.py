# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: obspy
#     language: python
#     name: obspy
# ---

# ### Prepare Data and Synthetics for Moment Tensor Inversion
#
# We have a few more steps to go through before inversion.
#
# One important step is to create synthetic Green's functions (aka synthetic seismograms) from wavenumber integration (FK). This type of method generates complete synthetic seismograms, we will use the software package ***Computer Programs in Seismology*** by R.B Herrmann.
# <a href="http://www.eas.slu.edu/eqc/eqccps.html" target="_blank">Link to software</a>
#
# In this notebook we will:
# * Filter and cut the seismograms
# * Calculate synthetic Green's functions
# * Filter the synthetic Green's functions

# Import third-party libraries
import os
from pathlib import Path
import pandas as pd
from obspy.core import read, UTCDateTime, Stream

# #### Data
# The final step of data preparation is to filter and cut the seismograms. After reading the processed SAC files into ObsPy we will:
#
# * Filter and taper data
# * Down-sample data to desired sampling interval
# * Cut data relative to a reference time (e.g. origin, first arrival, etc.)
# * Convert from meters to centimeters (tdmtpy units are in dyne-cm)
# * Save the final waveforms in SAC (required)
#
#
# Let's start by reading in the processed waveforms.

# +
evid = "40191336" # event of interest
event_dir = evid
infile = "%s/datetime.csv"%event_dir # we need the event origin time
station_file = "%s/station.csv"%event_dir

sacdir = "%s/sac"%event_dir # location of processed data
outdir = "%s"%event_dir # location of filtered/cut/down-sampled data for inversion
    
# Check if data directory exist
P = Path(sacdir)
if P.exists():
    # Read event info and station info into Pandas table
    df = pd.read_csv(infile,parse_dates=True)
    station_df = pd.read_csv("%s"%(station_file),parse_dates=True,dtype={"location":str},na_filter=False)
    
    origin_time = UTCDateTime(df["origin"][0])
    st = Stream()
    for _,row in station_df.iterrows():
        st += read("%s/%s.%s.%s.%s[%s]"%(
            sacdir,row.network,row.station,row.location,row.channel,row.component),format="SAC")
else:
    print("Directory %s does not exist. %s does not have instrument corrected data."%(sacdir,evid))
    
# -

# The next cell shows the processing parameters you need to define, you may need to change them for different events. Synthetic Green's functions must have the same filter, reduction velocity and sampling interval as the data.

# +
# Filter
freqmin = 0.02
freqmax = 0.05
corners = 3

# Desired sampling interval
dt = 1.0

# Reduction velocity in km/sec, 0 sets the reference time to origin time
vred = 0

# time before and after reference time, data will be cut before and after the reference time
time_before = 30
time_after = 200

# +
if vred:
    p = 1/vred
else:
    p = 0
    
st.filter("bandpass",freqmin=freqmin,freqmax=freqmax,corners=corners,zerophase=True)
st.taper(max_percentage=0.05)

# Trim and decimate the data
for tr in st:
    tr.decimate(factor=int(tr.stats.sampling_rate*dt), strict_length=False, no_filter=True)
    tr.resample(1/dt, strict_length=False, no_filter=True)
    tr.stats.sac.t1 = origin_time + p*(tr.stats.sac.dist) # set reference time
    tr.trim(tr.stats.sac.t1-time_before,tr.stats.sac.t1+time_after,pad=True,fill_value=0)
    tr.data = 100*tr.data # m/s to cm/s
    tr.stats.sac.b = -1*(origin_time - tr.stats.starttime)
    tr.stats.sac.o = 0
    # Save final trace using tdmtpy file name format
    sacout = "%s/%s.%s.dat"%(outdir,tr.id[:-4],tr.id[-1])
    tr.write(sacout,format="SAC")
# -

# ### Green's Functions
#
# Now is time to calculate the synthetic Green's functions
# * Execute the FK calculation
# * Apply the same filter to the synthetics
# * Save them to the appropriate format for inversion
#
# The FK calculation requires two input files, a velocity model file and a distance file. A velocity model file **gil7.d** is provided, this is a 1D model for northern California. We will create the distance file **dfile** from the Pandas table.

# +
model = "gil7"
#depths = round(df["depth"][0]) # Only compute GFs at catalog depth
depths = sorted([10,20,round(df["depth"][0])]) # compute GF at 10, 20 km and at catalog depth
npts = int(256) # number of points in the time series, must be a power of 2
t0 = int(0) # used to define the first sample point, t0 + distance_in_km/vred

# Location of synthetic Green's functions
green_dir = "%s/%s"%(event_dir,model)
Path(green_dir).mkdir(parents=True,exist_ok=True)
    
for depth in depths:
    # Create distance file
    dfile = ("{dist:.0f} {dt:.2f} {npts:d} {t0:d} {vred:.1f}\n")
    dfile_out = "%s/dfile"%event_dir
    with open(dfile_out,"w") as f:
        for _,row in station_df.iterrows():
            f.write(dfile.format(dist=row.distance,dt=dt,npts=npts,t0=t0,vred=vred))

    # Generate the synthetics
    os.system("hprep96 -M %s.d -d %s -HS %.4f -HR 0 -EQEX"%(model,dfile_out,depth))
    os.system("hspec96")
    os.system("hpulse96 -D -i > file96")
    os.system("f96tosac -B file96")

    # Filter and save the synthetic Green's functions
    greens = ("ZDD","RDD","ZDS","RDS","TDS","ZSS","RSS","TSS","ZEX","REX")

    for index,row in station_df.iterrows():      
        for j,grn in enumerate(greens):
            sacin = "B%03d%02d%s.sac"%(index+1,j+1,grn)
            sacout = "%s/%s.%s.%s.%.4f"%(green_dir,row.network,row.station,row.location,depth)
            tmp = read(sacin,format="SAC")
            tmp.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=corners,zerophase=True)
            tmp.write("%s.%s"%(sacout,grn),format="SAC") # overwrite

# Uncomment to remove unfiltered synthetic SAC files
os.system("rm B*.sac") # remove the unfiltered SAC files
# -

# ### Create input file for MTtime
# Now that we have prepared the data and synthetics for inversion, we can create the input file for tdmtpy.
# I will go over the input file format in the next notebook.

# +
# Create headers
headers = dict(datetime=df["origin"][0],
               longitude=df["lon"][0],
               latitude=df["lat"][0],
               depth=",".join([ "%.4f"%d for d in depths]),
               path_to_data=event_dir,
               path_to_green=green_dir,
               green="herrmann",
               components="ZRT",
               degree=5,
               weight="distance",
               plot=0,
               correlate=0,
              )

# Add station table
pd.options.display.float_format = "{:,.2f}".format
frame = {"station": station_df[["network","station","location"]].apply(lambda x: ".".join(x),axis=1)}
df_out = pd.DataFrame(frame)
df_out[["distance","azimuth"]] = station_df[["distance","azimuth"]]
df_out["ts"] = int(30)
df_out["npts"] = int(150)
df_out["dt"] = dt
df_out["used"] = 1
df_out[["longitude","latitude"]] = station_df[["longitude","latitude"]]
#print(df_out.to_string(index=False))
# -

# Save to file **mtinv.in**

# +
# write
with open("mtinv.in","w") as f:
    for key, value in headers.items():
        f.write("{0:<15}{1}\n".format(key,value))
    f.write(df_out.to_string(index=False))
    
# !cat mtinv.in
# -

# Now we can start the next tutorial and take a look at the moment tensor inversion package `mttime`
