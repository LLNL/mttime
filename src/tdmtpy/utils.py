# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
Utility functionality for tdmtpy
"""

import numpy as np
from scipy.signal import fftconvolve


def xcorr(data,template,normalize=True):
    """
    Compute cross-correlation coefficient

    :param data: first time series.
    :type data: :class:`numpy.ndarray`
    :param template: second time series.
    :type template: :class:`numpy.ndarray`
    :param normalize: If ``True`` normalizes correlations.
    :type normalize: bool
    :return: cross-correlation function.
    :rtype: :class:`numpy.ndarray`
    """
    c = fftconvolve(data,template[::-1],mode="valid")   
    if normalize:
        lent = len(template)
        # Zero-padding
        pad = len(c) - len(data) + lent
        pad1 = (pad+1) // 2
        pad2 = pad // 2
        data = np.hstack([np.zeros(pad1,dtype=data.dtype),
                          data,
                          np.zeros(pad2,dtype=data.dtype)])
        # Rolling sum
        window_sum = np.cumsum(data**2)
        np.subtract(window_sum[lent:],window_sum[:-lent],out=window_sum[:-lent])
        norm = window_sum[:-lent]
        
        norm *= np.sum(template ** 2)
        norm = np.sqrt(norm)
        mask = norm <= np.finfo(float).eps
        #    c[:] = 0
        if c.dtype == float:
            c[~mask] /= norm[~mask]
        else:
            c = c/norm
        c[mask] = 0
    return c


def gaussj(A,b):
    """
    Solves the linear matrix equation Ax = b

    Computes the solution of a well-determined linear matrix equation
    by performing Gaussian-Jordan elimination for a given matrix and
    transforming it to a reduced echelon form.

    :param A: coefficient matrix.
    :type A: :class:`numpy.ndarray`
    :param b: vector of dependent variables.
    :type b: :class:`numpy.ndarray`
    :return: solution vector x so that Ax = b.
    :rtype: :class:`numpy.ndarray`
    """
    # Make a copy to avoid altering input
    x = np.copy(b)
    n = len(x)

    # Gaussian-Jordan elimination
    for k in range(0,n-1):
        for i in range(k+1,n):
            if A[i,k] != 0.0:
                lam = A[i,k]/A[k,k]
                A[i,k+1:n] = A[i,k+1:n] - lam*A[k,k+1:n]
                x[i] = x[i]-lam*x[k]

    # Back substitution
    for k in range(n-1,-1,-1):
        x[k] = (x[k]-np.dot(A[k,k+1:n],x[k+1:n]))/A[k,k]

    return x


def az2baz(angle):
    """
    Azimuth to back-azimuth conversion

    :param angle: azimuth value in degrees between 0 and 360.
    :type angle: float or int
    :return: corresponding back-azimuth value in degrees.
    :rtype: float
    """
    if 0 <= angle <= 180:
        new_angle = angle + 180
    elif 180 < angle <= 360:
        new_angle = angle - 180
    else:
        raise ValueError("Input azimuth out of bounds: %s" % angle)

    return new_angle


def rotate_rt2ne(r, t, az):
    """
    Rotate to the great circle path.

    Rotates a pair of horizontal component data from north and east direction
    to radial and tangential direction.

    :param r: radial component data.
    :type r: :class:`~numpy.ndarray`
    :param t: tangential component data.
    :type t: :class:`~numpy.ndarray`
    :param az: The direction from a seismic source to the seismic station measured
        clockwise from north.
    :type az: float
    :return: rotated horizontal component data oriented in north and east direction.
    :rtype: Tuple of :class:`~numpy.ndarray`
    """
    ba = az2baz(az)
    ba = np.radians(ba)
    n = - r * np.cos(ba) + t * np.sin(ba)
    e = - r * np.sin(ba) - t * np.cos(ba)

    return n, e

def RGF_from_SW4(path_to_green=".", t0=0, file_name=None,
                 origin_time=None,event_lat=None,event_lon=None,depth=None,
                 station_name=None,station_lat=None,station_lon=None,
                 output_directory="sw4out"):
    """
    Function to convert reciprocal Green's functions from SW4 to tensor format
    
    Reads the reciprocal Green's functions (displacement/unit force) from SW4 and
    performs the summation to get the Green's function tensor.
    RGFs from SW4 are oriented north, east and positive down by setting az=0.
    
    Assumes the following file structure:
    f[x,y,z]/station_name/event_name.[x,y,z]
    
    :param path_to_green: path to RGFs.
    :type path_to_green: str
    :param t0: offset in time (>=0).
    :type t0: float
    :param file_name: name of RGF sac files from SW4 (event name).
    :type file_name: str
    :param origin_time: event origin time.
    :param event_lat: event latitude.
    :type event_lat: float
    :param event_lon: event longitude
    :type event_lon: float
    :param depth: source depth
    :type depth: float
    :param station_name: station names.
    :type station_name: list of strings.
    :param station_lat: station latitudes.
    :type station_lat: list of floats
    :param station_lon: station longitudes.
    :type station_lon: list of floats
    :param output_directory: output direcotry.
    :type output_directory: str
    """

    import os
    from obspy.core import read, Stream
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.core.util.attribdict import AttribDict
    
    # Defined variables (do not change)
    dirs = ["fz","fx","fy"] # directory to displacement per unit force
    du = ["duxdx","duydy","duzdz","duydx","duxdy","duzdx","duxdz","duzdy","duydz"]
    orientation = ["Z","N","E"] # set az=0 in SW4 so x=north, y=east
    cmpaz = [0,0,90]
    cmpinc = [0,90,90]

    # Create a new output directory under path_to_green
    dirout = "%s/%s"%(path_to_green,output_directory)
    if os.path.exists(dirout):
        print("Warning: output directory '%s' already exists."%dirout)
    else:
        print("Creating output directory '%s'."%dirout)
        os.mkdir(dirout)
        
    # Loop over each directory fx, fy, fz
    nsta = len(station_name)
    for i in range(3):
        # Set headers according to the orientation
        if dirs[i][-1].upper() == "Z":
            scale = -1 # change to positive up
        else:
            scale = 1
        
        # Loop over each station
        for j in range(nsta):
            station = station_name[j]
            stlo = station_lon[j]
            stla = station_lat[j]    
            dirin = "%s/%s/%s"%(path_to_green,dirs[i],station)
            print("Reading RGFs from %s:"%(dirin))
            st = Stream()
            for gradient in du:
                fname = "%s/%s.%s"%(dirin,file_name,gradient)
                st += read(fname,format="SAC")
    
            # Set station headers
            starttime = origin_time - t0
            dist, az, baz = gps2dist_azimuth(event_lat,event_lon,stla,stlo)

            # SAC headers
            sacd = AttribDict()
            sacd.stla = stla
            sacd.stlo = stlo
            sacd.evla = event_lat
            sacd.evlo = event_lon
            sacd.az = az
            sacd.baz = baz
            sacd.dist = dist/1000 # convert to kilometers
            sacd.o = 0
            sacd.b = -1*t0
            sacd.cmpaz = cmpaz[i]
            sacd.cmpinc = cmpinc[i]
            sacd.kstnm = station
        
            # Update start time
            for tr in st:
                tr.stats.starttime = starttime
                tr.stats.distance = dist
                tr.stats.back_azimuth = baz
        
            # Sum displacement gradients to get reciprocal Green's functions
            tensor = Stream()
            for gradient, element in zip(["duxdx","duydy","duzdz"],["XX","YY","ZZ"]):
                trace = st.select(channel=gradient)[0].copy()
                trace.stats.channel = "%s%s"%(orientation[i],element)
                tensor += trace
        
            trace = st.select(channel="duydx")[0].copy()
            trace.data += st.select(channel="duxdy")[0].data
            trace.stats.channel = "%s%s"%(orientation[i],"XY")
            tensor += trace
    
            trace = st.select(channel="duzdx")[0].copy()
            trace.data += st.select(channel="duxdz")[0].data
            trace.stats.channel = "%s%s"%(orientation[i],"XZ")
            tensor += trace
    
            trace = st.select(channel="duzdy")[0].copy()
            trace.data += st.select(channel="duydz")[0].data
            trace.stats.channel = "%s%s"%(orientation[i],"YZ")
            tensor += trace
            
            # Set sac headers before saving
            print("    Saving GFs to %s"%dirout)
            for tr in tensor:
                tr.trim(origin_time, tr.stats.endtime)
                tr.data = scale*tr.data
                tr.stats.sac = sacd
                sacout = "%s/%s.%.4f.%s"%(dirout,station,depth,tr.stats.channel)
                #print("Writing %s to file."%sacout)
                tr.write(sacout,format="SAC")
