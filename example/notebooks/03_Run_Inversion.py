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
#     display_name: ncev
#     language: python
#     name: ncev
# ---

# ## Moment Tensor Inversion using `tdmtpy`
#
# Now we can start the inversion using the input file we created.
#

# +
import tdmtpy
# %matplotlib inline

from IPython.display import Image # to display image files inside the Jupyter notebook
# -

# #### Let's take a look at mtinv.in
#
# ```
# origin         2019-07-16T20:11:01.470000Z
# longitude      -121.7568
# latitude       37.8187
# depth          10.0000,12.0000,20.0000
# path_to_data   40191336
# path_to_green  40191336/gil7
# green          herrmann
# components     ZRT
# degree         5
# weight         distance
# plot           1
# correlate      0
#     station  distance  azimuth  ts  npts   dt  used  longitude  latitude
#  BK.QRDG.00     80.99   335.29  30   150 1.00     1    -122.14     38.48
#  BK.RUSS.00     81.16   353.18  30   150 1.00     1    -121.87     38.54
#   BK.CVS.00     84.88   313.73  30   150 1.00     1    -122.46     38.35
#  BK.OAKV.00     88.89   320.02  30   150 1.00     1    -122.41     38.43
#  BK.MCCM.00    105.12   290.48  30   150 1.00     1    -122.88     38.14
#  BK.FARB.00    110.46   263.41  30   150 1.00     1    -123.00     37.70
#  BK.WELL.00    113.71    52.46  30   150 1.00     1    -120.72     38.44
#   BK.SAO.00    120.23   166.71  30   150 1.00     1    -121.45     36.76
#   BK.CMB.00    122.83    78.33  30   150 1.00     1    -120.39     38.03
#  BK.MNRC.00    132.06   333.21  30   150 1.00     1    -122.44     38.88
#   BK.SCZ.00    139.07   166.84  30   150 1.00     1    -121.40     36.60
#  BK.BUCR.00    142.56    96.01  30   150 1.00     1    -120.15     37.67
# ```

# The first twelve lines are keyword arguments for constructing the Configure object,
# which sets up the inversion based on the inputs you provided (e.g. source depths, types of Green's functions, etc.).
# Do not change the keywords in the first column, only the parameters in the second column.
# The first three lines refer to the event origin time, longitude and latitude, they are not used and can be left blank.
#
# |        | keywords      | description                       | options and/or type |
# | ------ | ------------- | --------------------------------- | ------------------- |
# | Line1  | origin        | event origin time, optional       | str                 |
# | Line2  | longitude     | event longitude, optional         | float               |
# | Line3  | latitude      | event latitude, optional          | float               |
# | Line4  | depth         | comma-delimited source depths     | str                 |
# | Line5  | path_to_data  | directory of processed data       | str                 |
# | Line6  | path_to_green | directory of processed synthetics | str                 |
# | Line7  | green         | synthetic Green's function format | herrmann or tensor  |
# | Line8  | components    | data and synthetic components     | Z or ZRT            |
# | Line9  | degree        | deviatoric or full inversion      | 5 or 6              |
# | Line10 | weight        | weighting methods: none, inverse distance or inverse variance | none, distance, or variance
# | Line11 | plot          | flag to turn on/off plotting          | 1 or 0
# | Line12 | correlate     | cross-correlate data and synthetics to estimate best time shift | 1 or 0
#

# Lines 13 and onward contain station information, line 13 is the station header and should not be modified.
#
# | headers   | description                                       |
# | --------- | ------------------------------------------------- |
# | station   |  file names of data and synthetics                |
# | distance  | source-receiver distance                          |
# | azimuth   | source-receiver azimuth                           |
# | ts        | shift data by the number of time points specified |
# | npts      | number of samples to invert                       |
# | dt        | sampling interval                                 |
# | used      | specify which components to invert, set 1 to invert and 0 for prediction only.<br>For three component data you can set flags for individual components, e.g 110 will invert ZR components only. |
# | longitude | station longitude                                 |
# | latitude  | station latitude                                  |

# +
# Call the Configure object to read the input file and set up the inversion
config = tdmtpy.Configure(path_to_file="mtinv.in")

# Quick look at the attributes
print(config)
# -

# Pass the parameters to the Inversion object and launch the inversion
# The default is to plot all solutions
tdmt = tdmtpy.Inversion(config=config)
tdmt.invert()

# You can display the png image files here
Image(filename="bbwaves.d10.0000.00.png")
#Image(filename="bbwaves.d10.0000.01.png")



# Or turn on interactive display when you call the plotting function
tdmt.plot(show=True)

# Or plot the results as a function of source depth
tdmt.plot(view="depth",show=True)

# +
# Finally save the results to file, by default it will create two files: config.out and d{depth}mtinv.out
# Default is to save all the results
tdmt.write()
# !cat d12.0000.mtinv.out

# Setting option to 'preferred' will only save the best solution
tdmt.write(option="preferred")
# -

# # Find the best solution
# Make some changes to your input file, such as changing the time shifts, removing bad stations, etc. to get a better solution.


