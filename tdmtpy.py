# SPDX-License-Identifier: (BSD-3)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python module for Time Domain Moment Tensor Inversion (tdmtpy), Version: 0.1

Last updated on February 11, 2020

@module: tdmtpy
@author: Andrea Chiang (chiang4@llnl.gov)

Usage
1. From command line: tdmt [input_file], if no input file is specified code will look for default input file "./mtinv.in"
2. Input file example (e.g. mtinv.in), check :Class: `tdmtpy.Header` and :Class: `tdmtpy.Station` for more details on the input parameters.

datetime:   2019-07-16T20:10:31.473
longitude:  -121.757
latitude:   37.8187
data_dir:   example/dc
green_dir:  example/gf
greentype:  herrmann
component:  ZRT
depth:      10
degree:     5
weight:     1
plot:       1
correlate:  0
NAME  DISTANCE  AZIMUTH  SHIFT  NPTS  DT  USED(1/0)  FILTER  NC  NP  LCRN  HCRN  MODEL  STLO  STLA
BK.FARB.00  110 263 30 100 1.0  1      bp 2 2 0.05 0.1  gil7  -123.0011   37.69782
BK.SAO.00   120 167 30 150 1.0  1      bp 2 2 0.05 0.1  gil7  -121.44722  36.76403
BK.CMB.00   123  78 30 150 1.0  0      bp 2 2 0.05 0.1  gil7  -120.38651  38.03455
BK.MNRC.00  132 333 30 150 1.0  1      bp 2 2 0.05 0.1  gil7  -122.44277  38.87874
NC.AFD.     143  29 30 150 1.0  1      bp 2 2 0.05 0.1  gil7  -120.968971 38.94597

3. Data and Green's functions are in SAC binary format, and are corrected for instrument response,
filtered, and decimated. File name coventions are described below:
Data - [NAME].[COMPONENTS].dat
		NAME = station name in inputfile
		COMPONENTS = t, r, z
		e.g. BK.CMB.00.z.dat, BK.CMB.00.t.dat
GFs - [NAME].[DEPTH].[GF_NAME]
		NAME = station name in inputfile
		DEPTH = source depth with four significant digits
		COMPONENTS = t, r, z
		GF_NAME = herrmann format has 10: tss tds rss rds rdd zss zds zdd rex zex, e.g. BK.CMB.00.10.0000.zds
			  tensor format has 18 (if using all three components): xx, yy, zz, xy, xz, yz, e.g. BK.CMB.00.10.0000.zxy
4. Two output files are created "mtinv.out" and "max.mtinv.out" after running the code.
	mtinv.out = moment tensor depth search results, best solution on the second line (after header)
	max.mtinv.out = best solution with the highest VR, includes additional station information
5. If plot = 1 code will generate figures (e.g. figure0.pdf, figure1.pdf, etc.) with beach balls and waveform fits plotted
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from obspy.core import read, Stream
from scipy.signal import fftconvolve
import sys

import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.collections as mpl_collections
from matplotlib import patches, transforms
from obspy.imaging.beachball import xy2patch
from obspy.imaging.scripts.mopad import BeachBall as mopad_BeachBall
from obspy.imaging.scripts.mopad import MomentTensor as mopad_MomentTensor
from obspy.imaging.scripts.mopad import epsilon

from math import pi, sqrt

def loadfile(path_and_file=None):
    """
    Input text file containing all information needed to set :class:`tdmtpy.Header` and
    :class:`tdmtpy.Station` attributes.
    """
    if path_and_file is None:
        fname = 'mtinv.in'
    else:
        fname = path_and_file  
    
    try:
        with open(fname) as f: pass
    except IOError:
        raise IOError("Input file '%s' not found."%fname)
        
    # `Header` names and corresponding types        
    parser = dict(datetime = str,
                  longitude = float,
                  latitude = float,
                  datadir = str,
                  greendir = str,
                  greentype = lambda x: x.upper(),
                  component = lambda x: x.lower(),
                  depth = _set_depth,
                  degree = int,
                  weight = int,
                  plot = int,
                  correlate = int)

    # Read headers and define formats
    with open(fname,'r') as f:
        for key, parse in parser.items():
            parser[key] = parse(next(f).split()[1])
        next(f) # Skip Line 14
        items = [ line.split() for line in f ] # Station-specific info 
    header = Header(parser)
    
    # `Station` names and corresponding types            
    parser = dict(name = np.object_,
                  distance = np.float_,
                  azimuth = np.float_,
                  shift = np.int_,
                  npts = np.int_,
                  dt = np.float_,
                  used = np.object_,
                  filtertype = np.object_,
                  nc = np.int_,
                  np = np.int_,
                  lcrn = np.float_,
                  hcrn = np.float_,
                  model = np.object_,
                  stlo = np.float_,
                  stla = np.float_)
    for (key, parse), col in zip(parser.items(),zip(*items)):
        parser[key] = parse(col)
    parser['name'] = parser['name'].astype('U')
    parser['used'] = parser['used'].astype('U')
    parser['filtertype'] = parser['filtertype'].astype('U')
    parser['model'] = parser['model'].astype('U')
    station = Station(parser,ncomp=len(header.component))
    
    mt = TDMT(header=header,station=station)
    
    return mt

def _set_depth(depth_str):
    """
    Depth search based on minimum source depth, maximum source depth, and vertical spacing
    
    :param depth_str: a comma-delimited string depth_min,depth_max,spacing (e.g. 10,30,5) or a fixed depth
    """
    depth = np.float_(depth_str.split(':'))
    if depth.size == 3:
        start = depth[0]
        step = depth[2]
        stop = depth[1] + step
        depth_profile = [ z for z in np.arange(start,stop,step) ]
    elif depth.size == 1:
        depth_profile = depth
    else: raise ValueError
        
    return depth_profile

class Header(object):
    """
    Container for header information of tdmtpy TDMT object
    
    Header object contains all header information of a :class:`tdmtpy.TDMT` object. These are
    required for every TDMT object.
    
    :param hdr: dict, optional
        dict containing headers loaded from :func: `tdmtpy.loadfile`
        
    .. rubric:: Attributes, defaults to None if not supplied. All attributes must be defined to run inversion.
    datetime: string
        Event origin time in ObsPy UTCDatetime calendar/ordinal date representation 
        (e.g. "2009-W53-7T12:23:34.5" or "2009-365T12:23:34.5" )
    longitude: float
        Event longitude in decimal degrees
    latitude: float
        Event latitude in decimal degrees
    datadir: string
        Path to data files
    greendir: string
        Path to green's function files
    greentype: string
        Green's function format. Supported formats are: "HERRMANN"-1D elementary seismograms,
        "TENSOR"-1D or 3D Green's tensor.
    component: string
        Data and Green's function components. Options are: "ZRT"-vertica, radial and transverse,
        "Z"-vertical only.
    depth: np.float
        Fixed source depth or vertical depth profile on which the inverse is solved.
        Depth profile is constructed using the parameters speicifed in the input file.
        See Class Header.set_depth for more details.
    degree: int
        Number of independent parameters in the inversion. Options are: 5-deviatoric inversion,
        6-full inversion.
    weight: int
        Data weighting function, same weights are applited to all components of a station.
        Options are: 0-No weights applied, 1-Inverse distance weighting, 2-inverse variance weighting
    plot: int
        Whether or not to plot results. False/0-No, True/1-Yes
    correlate: int
        Whether or not to search for best time shifts between data and synthetics using cross-correlation.
        False/0-No, True/1-Yes.
    """
    def __init__(self,hdr=None):
        if hdr is None:
            self.datetime = None
            self.longitude = None
            self.latitude = None
            self.datadir = None
            self.greendir = None
            self.greentype = None
            self.component = None
            self.depth = None
            self.degree = None
            self.weight = None
            self.plot = False
            self.time_search = 0
            self.correlate = 0
        else:
            self._read_hdr(hdr)

    def __str__(self):
        f = "{0:>15}: {1}\n"
        ret = ''.join([ f.format(key,str(getattr(self,key))) for key in vars(self)] )
        return ret
    
    def _read_hdr(self,hdr):
        self.datetime = hdr['datetime']
        self.longitude = hdr['longitude']
        self.latitude = hdr['latitude']
        self.datadir = hdr['datadir']
        self.greendir = hdr['greendir']
        self.greentype = hdr['greentype']
        self.component = hdr['component']
        self.depth = hdr['depth']
        self.degree = hdr['degree']
        self.weight = hdr['weight']
        self.plot = hdr['plot']
        self.correlate = hdr['correlate']

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
        
class Station(object):
    """
    Container for station information of tdmtpy TDMT object
    
    Station object contains all station information of a :class:`tdmtpy.TDMT` object. These are
    required for every TDMT object.
    
    :param stat: dict, optional
        A dictionary containing station-specific parameters id (file name), distance, azimuth,
        dt, npts, shift, used. See README on format.
    :param ncomp: int, required
        number of components
    
    .. rubric:: Attributes, required unless marked as optional    
    nsta: int
        Number of stations
    ncomp: int
        Numbber of components
    name: np.string
        Station/file names
    distance: np.float
        Source-receiver distance
    azimuth: np.float
        Source-receiver azimuth
    dt: np.float
        Sampling interval
    npts: np.int
        Number of samples
    shift: np.int
        Time shift in samples
    used: np.int
        Invert data (1) or prediction only (0)
    index1: np.int
        Beginning of each component/trace in d = Gm
    index2: np.int
        End of each component/trace in d = Gm
    
    """
    def __init__(self,stat=None,ncomp=None):
        if stat is None:
            self.nsta = None
            self.ncomp = None
            self.name = None
            self.distance = None
            self.azimuth = None
            self.shift = None            
            self.npts = None
            self.dt = None
            self.used = None
            self.index1 = None
            self.index2 = None
        else:
            self._read_stat(stat,ncomp)

    def __str__(self):
        f = "{0:>10}: {1}\n"
        ret = ''.join([ f.format(key,str(getattr(self,key))) for key in vars(self)] )
        return ret

    def _read_stat(self,stat,ncomp):
        self.nsta = len(stat['name'])
        self.ncomp = ncomp
        self.name = stat['name']
        self.distance = stat['distance']
        self.azimuth = stat['azimuth']
        self.shift = stat['shift']
        self.npts = stat['npts']
        self.dt = stat['dt']
        self._set_used(stat['used'])
        self._set_indices()

    def _set_used(self,used):
        """
        Determine which components to invert
        """
        if self.ncomp == 1:
            self.used = np.ones((self.nsta,1),dtype=np.int)
            self.used.T[:] = used
        else:
            self.used = np.ones((self.nsta,3),dtype=np.int)
            if np.any(np.char.rfind(used,',') != -1):
                for i in range(self.nsta):
                    self.used[i,:] = used[i].split(',')
            else:
                self.used.T[:] = used
    
    def _set_indices(self):            
        """
        Calculate indices of each component in inversion matrix (d and G)
        """
        if self.npts is None or self.ncomp is None or self.nsta is None:
            print('Missing station information.')
        else:
            index2 = np.cumsum(np.repeat(self.npts,self.ncomp), dtype=np.int)
            index1 = np.zeros(self.ncomp * self.nsta, dtype=np.int)
            index1[1::] = index2[0:-1]
            self.index1 = index1.reshape(self.nsta,self.ncomp)
            self.index2 = index2.reshape(self.nsta,self.ncomp)
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
        
def xcorr(arr1,arr2,normalize=True):
    """
    Compute cross-correlation coefficient
    """
    c = fftconvolve(arr2[::-1],arr1,mode='valid')       
    if normalize:
        norm = (np.sum(arr1**2)*np.sum(arr2**2))**0.5
        if norm <= np.finfo(float).eps: # norm is zero
            c[:] = 0
        elif c.dtype == float:
            c /= norm
        else:
            c = c/norm

    return c

def invert(d,G,w,greentype,degree,plot,nsta,used,index1,index2):
    
    # d = data    
    a = invert4a(d,G,w)
    M = a2M(a,greentype,degree)
    
    # Variance reduction
    Gm = np.dot(G,a)
    dGm = w*(d-Gm)**2
    dd = w*d**2
    VR = (1-np.sum(dGm)/np.sum(dd))*100

    # Station VR          
    used = np.sum(used,axis=1)
    ind1 = index1[:,0]
    ind2 = index2[:,-1]
    staVR = [ (1-np.sum(dGm[b:e])/np.sum(dd[b:e]))*100
             if yes else None for yes, b, e in zip(used,ind1,ind2) ]
    
    out = {}
    out.update(decompose(M,plot))
    if plot:
        out['a'] = a # a coefficients to compute synthetics
    out['VR'] = VR
    out['staVR'] = staVR
    
    # Display outputs        
    iso = {5:'Deviatoric',6:'Full'}
    print('\n%s Moment Tensor Inversion'%iso[degree])
    print('Mw = %.2f'%out['mw'])
    print('Percent DC/CLVD/ISO = %d/%d/%d'%(out['pdc'],out['pclvd'],out['piso']))
    print('VR = %.2f%%'%out['VR'])
    
    return out

def gaussj(A,b):    
    """
    Gaussian-Jordan elimination and back substitution
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

def invert4a(d,G,w):
    # Perform linear least-squares inversion
    # coefficients to elementary moment tensors
    d = d[w!=0]
    G = G[w!=0,:] # squeeze, remove single-dimensional entries
    w = w[w!=0]
    
    Gt = (G).transpose()
    Gtd = np.dot(Gt,np.dot(np.diag(w),d))
    GtG = np.dot(Gt,np.dot(np.diag(w),G))
        
    # Compute the inverse GtGinv
    a = gaussj(GtG,Gtd)
    
    return a

def a2M(a,greentype,degree):
    # aij coefficients (weights to elementary moment tensors) to moment tensor elements

    M = [ None for _ in range(6) ]
    
    # Ordering Mxx, Myy, Mzz, Mxy, Mxz, Myz
    
    if greentype == 'TENSOR':
        M[3] = a[0]
        M[4] = a[3]
        M[5] = a[2]
        if degree == 6:
            M[0] = a[1] - a[4] + a[5]
            M[1] = -a[1] + a[5]
            M[2] = a[4] + a[5]
        elif degree == 5:
            M[0] = a[1] - a[4]
            M[1] = -a[1]
            M[2] = a[4]
        else: pass

    if greentype == 'HERRMANN':
        M[0] = a[0]
        M[1] = a[3]
        M[3] = a[1]
        M[4] = a[2]
        M[5] = a[4]
        if degree == 6:
            M[2] = a[5]
        elif degree == 5:
            M[2] = -(a[0]+a[3])
        else: pass
            
    return M

def find_strike_rake_dip(u,n):  
    """
    Compute strike,rake and dip
    """   
    # Inputs: u = slip vector, n = fault normal
    dip = np.arccos(-1*u[2])*180/np.pi
    strike = np.arcsin(-1*u[0]/np.sqrt(u[0]**2+u[1]**2))*180/np.pi
    # Determine the quadrant
    if u[1] < 0:
        strike = 180-strike

    rake = np.arcsin(-1*n[2]/np.sin(dip*np.pi/180))*180/np.pi
    cos_rake = n[0]*np.cos(strike*np.pi/180)+n[1]*np.sin(strike*np.pi/180)
    if cos_rake < 0:
        rake = 180-rake
    if strike < 0:
        strike = strike+360

    if rake < -180:
        rake = rake+360
    if rake > 180:
        rake = rake-360
        
    return (strike,rake,dip)

def find_fault_planes(M,M_dc):
    eigVal, eigVec = np.linalg.eig(M_dc)
        
    # sort in ascending order:
    dc_eigvec = np.real(np.take(eigVec,np.argsort(eigVal),1))
        
    # Principal axes:
    #n = dc_eigvec[:,1]
    p = dc_eigvec[:,2]
    t = dc_eigvec[:,0]

    # str/rake/dip for plane-1
    u1 = (1/np.sqrt(2))*(t+p) # slip vector
    n1 = (1/np.sqrt(2))*(p-t) # fault normal
        
    # u,n calculations from Vavrycuk (2015) angles.m
    if u1[2] > 0:  # vertical component is always negative!
        u1 = -1*u1
    if (np.dot(np.dot(u1.T,M),n1)+np.dot(np.dot(n1.T,M),u1)) < 0:
        n1 = -1*n1
    str1,rake1,dip1 = find_strike_rake_dip(u1,n1)
            
    # str/rake/dip for plane-2
    u2 = n1
    n2 = u1
    if u2[2] > 0: # vertical component is always negative!
        u2 = -1*u2
        n2 = -1*n2
    str2,rake2,dip2 = find_strike_rake_dip(u2,n2)
        
    #null_axis = dc_eigvec[:,1]
    #t_axis = t
    #p_axis = p
    
    return ( [(str1,dip1,rake1),(str2,dip2,rake2)] )

def eigen2lune(lam):
    """
    Convert eigenvalues to source-type parameters on a lune (Tape and Tape, GJI 2012)
    :param eigenvalues: in descending order
    """
    lambda_mag = sqrt(lam[0]**2 + lam[1]**2 + lam[2]**2)
        
    if np.sum(lam) != 0:
        bdot = np.sum(lam)/(sqrt(3)*lambda_mag)
        if bdot > 1:
            bdot = 1
        elif bdot < -1:
            bdot = -1
        #bdot[bdot>1] = 1
        #bdot[bdot<-1] = -1
        delta = 90 - np.arccos(bdot)*180/pi
    else:
        delta = 0.
            
    if lam[0] == lam[2]:
        gamma = 0
    else:
        gamma = np.arctan((-lam[0] + 2*lam[1] - lam[2]) / (sqrt(3)*(lam[0] - lam[2])))*180/pi
        
    return ([gamma,delta])
        
def decompose(m,plot=False):
    M = np.array([[m[0],m[3],m[4]],
                  [m[3],m[1],m[5]],
                  [m[4],m[5],m[2]]])
    M *= 1e20 # dyne-cm
    # Isotropic part:
    M_iso = np.diag(np.array([1. / 3 * np.trace(M),
                              1. / 3 * np.trace(M),
                              1. / 3 * np.trace(M)]))
    miso = 1. / 3 * np.trace(M)
        
    # Deviatoric part:
    M_dev = M - M_iso
        
    # compute eigenvalues and -vectors:
    eigVal, _ = np.linalg.eig(M)
    eigenvalues = np.real(np.take(eigVal,np.argsort(eigVal)[::-1]))
        
    # deviatoric eigenvalues and -vectors
    eigVal, eigVec = np.linalg.eig(M_dev)
        
    # sort in absolute value ascending order:
    dev_eigval = np.real(np.take(eigVal,np.argsort(abs(eigVal))))
    dev_eigvec = np.real(np.take(eigVec,np.argsort(abs(eigVal)),1))

    # Jost and Herrmann, 1989 definition of eigenvalues:
    m1 = dev_eigval[0]
#   m2 = dev_eigval[1]
    m3 = dev_eigval[2] # deviatoric moment
      
    if m3 == 0.: # isotropic only
        F = 0.5
    else:
        F = -1*m1/m3
        
    # Construct Dyadic Description of Vector Dipoles:
    a3a3 = np.column_stack([dev_eigvec[:,2],dev_eigvec[:,2],dev_eigvec[:,2]])
    a3a3 = a3a3*a3a3.T
    a2a2 = np.column_stack([dev_eigvec[:,1],dev_eigvec[:,1],dev_eigvec[:,1]])
    a2a2 = a2a2*a2a2.T
    a1a1 = np.column_stack([dev_eigvec[:,0],dev_eigvec[:,0],dev_eigvec[:,0]])
    a1a1 = a1a1*a1a1.T
        
    M_clvd = m3*F*(2*a3a3-a2a2-a1a1) # CLVD tensor
    mclvd = abs(m3)*abs(F)*2
        
    M_dc = m3*(1-2*F)*(a3a3-a2a2) # DC tensor
    mdc = abs(m3)*abs(1-2*F)
        
    # Bowers and Hudson moment
    mo = abs(miso)+abs(m3) # iso moment + dev moment
    mw = (np.log10(mo)-16.05)*2/3
    mw_dev = 2*np.log10(mo)/3-10.73

    # Calculate percentage
    #piso = int(round(abs(miso)/mo*100,6))
    #pdc = int(round((1-2*abs(F))*(1-piso/100.)*100,6))
    #pclvd = int(100 - piso - pdc)
    piso = abs(miso)/mo*100
    pdc = mdc/mo*100
    pclvd = mclvd/mo*100
        
    # DC Fault planes
    fps =  find_fault_planes(M,M_dc)
    
    # Find gamma and delta, Tape&Tape lune parameters
    lune = eigen2lune(eigenvalues)
    

    res = {'M':[ M[0,0],M[1,1],M[2,2],M[0,1],M[0,2],M[1,2] ],
           'eigenvalues':eigenvalues, 'mo':mo, 'mw':mw, 'mw_dev':mw_dev,
           'miso':miso, 'mdc':mdc, 'mclvd': mclvd,
           'pdc':pdc, 'pclvd':pclvd,'piso':piso, 'fps':fps, 'lune':lune}
    if plot:
        DEV = [ M_dev[0,0],M_dev[1,1],M_dev[2,2],M_dev[0,1],M_dev[0,2],M_dev[1,2] ]
        ISO = [ M_iso[0,0],M_iso[1,1],M_iso[2,2],M_iso[0,1],M_iso[0,2],M_iso[1,2] ]
        CLVD = [ M_clvd[0,0],M_clvd[1,1],M_clvd[2,2],M_clvd[0,1],M_clvd[0,2],M_clvd[1,2] ]
    
        res.update({'DEV':DEV, 'ISO':ISO, 'CLVD':CLVD})
    
    return res
              
def new_page(nsta,nrows,ncols,annot='',offset=2,figsize=(10,8)):
    gs = GridSpec(nrows+offset,ncols,hspace=0.5,wspace=0.1)
    f = plt.figure(figsize=figsize)
    
    # Annotations and beach balls
    ax0 = f.add_subplot(gs[0:offset,:],xlim=(-5,3.55),ylim=(-0.75,0.6),aspect='equal')
    ax0.text(-5,0,annot,fontsize=11,verticalalignment='center')
    ax0.set_axis_off()

    # Waveforms
    ax1 = np.empty((nsta,3),dtype=np.object) # create empty axes
    for i in range(nsta):
        ax1[i,0] = f.add_subplot(gs[i+offset,0])
        ax1[i,1] = f.add_subplot(gs[i+offset,1])
        ax1[i,2] = f.add_subplot(gs[i+offset,2])

    # Adjust axes

    for i in range(nsta-1):
        adjust_spines(ax1[i,0],['left','bottom'])
        for j in range(1,3):
            adjust_spines(ax1[i,j],[])      
    adjust_spines(ax1[-1,0],['left','bottom'])
    adjust_spines(ax1[-1,1],['bottom'])
    adjust_spines(ax1[-1,2],['bottom'])
    
    # Title
    ax1[0,0].set_title('Vertical',verticalalignment='bottom',fontsize=10)
    ax1[0,1].set_title('Radial',verticalalignment='bottom',fontsize=10)
    ax1[0,2].set_title('Tangential',verticalalignment='bottom',fontsize=10)
    
    return (f,ax0,ax1)

def adjust_spines(ax,spines):
    ax.tick_params(direction='in',labelsize=8)
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',5)) # outward by 5 points
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('time (s)')
    else:
        ax.xaxis.set_ticks([])

def beach(fm, linewidth=1, facecolor='0.75', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
          zorder=100, mopad_basis='NED', axes=None, show_iso=False):
    """
    Function taken from obspy.imaging.mopad_wrapper, minor modification to
    include isotropic components, original source code only handles pure iso.
    """
    # initialize beachball
    mt = mopad_MomentTensor(fm, system=mopad_basis)
    bb = mopad_BeachBall(mt, npoints=size)

    ## Snippets added by A. Chiang
    if show_iso:
        bb._plot_isotropic_part = True
        bb._nodallines_in_NED_system()
    
    ##
    bb._setup_BB(unit_circle=False)

    # extract the coordinates and colors of the lines
    radius = width / 2.0
    neg_nodalline = bb._nodalline_negative_final_US
    pos_nodalline = bb._nodalline_positive_final_US
    tension_colour = facecolor
    pressure_colour = bgcolor

    if nofill:
        tension_colour = 'none'
        pressure_colour = 'none'

    # based on mopads _setup_plot_US() function
    # collect patches for the selection
    coll = [None, None, None]
    coll[0] = patches.Circle(xy, radius=radius)
    coll[1] = xy2patch(neg_nodalline[0, :], neg_nodalline[1, :], radius, xy)
    coll[2] = xy2patch(pos_nodalline[0, :], pos_nodalline[1, :], radius, xy)

    # set the color of the three parts
    fc = [None, None, None]
    if bb._plot_clr_order > 0:
        fc[0] = pressure_colour
        fc[1] = tension_colour
        fc[2] = tension_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = tension_colour
            if bb._plot_curve_in_curve < 1:
                fc[1] = pressure_colour
                fc[2] = tension_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = pressure_colour
                fc[2] = tension_colour
    else:
        fc[0] = tension_colour
        fc[1] = pressure_colour
        fc[2] = pressure_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = pressure_colour
            if bb._plot_curve_in_curve < 1:
                fc[1] = tension_colour
                fc[2] = pressure_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = tension_colour
                fc[2] = pressure_colour

    if bb._pure_isotropic:
        if abs(np.trace(bb._M)) > epsilon:
            # use the circle as the most upper layer
            coll = [coll[0]]
            if bb._plot_clr_order < 0:
                fc = [tension_colour]
            else:
                fc = [pressure_colour]

    # transform the patches to a path collection and set
    # the appropriate attributes
    collection = mpl_collections.PatchCollection(coll, match_original=False)
    collection.set_facecolors(fc)
    # Use the given axes to maintain the aspect ratio of beachballs on figure
    # resize.
    if axes is not None:
        # This is what holds the aspect ratio (but breaks the positioning)
        collection.set_transform(transforms.IdentityTransform())
        # Next is a dirty hack to fix the positioning:
        # 1. Need to bring the all patches to the origin (0, 0).
        for p in collection._paths:
            p.vertices -= xy
        # 2. Then use the offset property of the collection to position the
        # patches
        collection.set_offsets(xy)
        collection._transOffset = axes.transData
    collection.set_edgecolors(edgecolor)
    collection.set_alpha(alpha)
    collection.set_linewidth(linewidth)
    collection.set_zorder(zorder)

    return collection
     
class TDMT(object):    
    """
    Class for time domain moment tensor inversion, the necessary attributes need to be defined
    by reading the input parameter file (e.g. mtinv.in)
    """
    def __init__(self,header=None,station=None):    
        
        self.header = header
        self.station = station
        if self.header is not None and self.station is not None:
            self._read_green_library() # Read Green's functions (synthetics)
            data = self._read_data() # Read data
            if self.header.correlate:
                self._find_time_shift(data)
                d = [ self._data2d(data,shift) for shift in self.shift ]   
            else:
                d = self._data2d(data,self.station.shift)
                d = [  d for _ in self.header.depth ] # data array for inversion
                self.shift = [ self.station.shift for _ in self.header.depth ]
        
            self.d = d
            self._calculate_weights()
        else:
            print('No headers are defined.')
    
    def run(self):
        # Run inversion
        self._run_inversion()
        # Find solution with maximum VR
        val = [ r.get('VR') for r in self.solutions ]
        self.best = val.index(max(val)) # Store index of best solution
        self.station.shift = self.shift[self.best] # update sample shift
        self.d = self.d[self.best] # update data vector
        if self.header.weight == 2: # update weights if using inverse variance
            self.stawei = self.stawei[self.best]
        else:
            self.stawei = self.stawei[0]
    
    def write(self):
        # Write solutions to files
        self._write_solutions2list() # depth search results, best solution with max VR on top
        self._write_best2text() # write station-specific results for best solution
        if self.header.plot:
            self.plot()
            
    def plot(self):
        # Plot waveform fits and focal mechanisms of best solution
        solution = self.solutions[self.best]
        Gm = np.dot(self.G[self.best],solution['a'])
        stavr = [ '%.0f'%val if val else '' for val in solution['staVR'] ]
        #time = self.npts*self.dt
        annot = '\n'.join(['Depth = %s km'%self.header.depth[self.best],
                'Mo = %.2E dyne-cm'%solution['mo'],
                'Mw = %.2f'%solution['mw'],
                'Percent DC/CLVD/ISO = %d/%d/%d'%(solution['pdc'],solution['pclvd'],solution['piso']),
                'VR = %.2f%%'%solution['VR']])
                
        # Beach balls TURN 2 FUNCTION
        if self.header.degree == 5:
            fm = (solution['M'], solution['fps'][0], solution['CLVD'])
            width = (1, 0.01*solution['pdc'],0.01*solution['pclvd'])
            fmtxt = ('Deviatoric','DC','CLVD')
            fmx = (1-0.01*solution['pdc']*0.25, 2.5-0.01*solution['pclvd']*0.5)
        elif self.header.degree == 6:
            fm = (solution['M'],solution['DEV'],solution['ISO'])
            width = (1, 0.01*(solution['pdc']+solution['pclvd']), 0.01*solution['piso'])
            fmtxt = ('Full','Deviatoric','ISO')
            fmx = (1-0.01*(solution['pdc']+solution['pclvd'])*0.25, 2.5-0.01*solution['piso']*0.5)

        nrows = 6
        ncols = 3
        a = self.station.nsta/nrows
        nPages = int(a) + ((int(a) - a) !=0 )
        lst = list(range(0,self.station.nsta,nrows))
        lst.append(self.station.nsta)
        pages = (range(lst[i],lst[i+1]) for i in range(nPages))
            
        x = 0.55*np.sin(self.station.azimuth*np.pi/180)
        y = 0.55*np.cos(self.station.azimuth*np.pi/180)
        syntcol = np.empty(self.station.used.shape,dtype='<U5')
        syntcol[self.station.used==1] = 'green'
        syntcol[self.station.used==0] = '0.5'
        datacol = np.empty(self.station.used.shape,dtype='<U5')
        datacol[self.station.used==1] = 'black'
        datacol[self.station.used==0] = '0.5'
        for page, group in enumerate(pages):
            #tmax = np.max(time[group])
            f, ax0, ax1 = new_page(len(group),nrows+1,ncols,annot=annot) #+1 for beach ball
            ax0.text(fmx[0],0,'=',horizontalalignment='center',verticalalignment='center')
            ax0.text(fmx[1],0,'+',horizontalalignment='center',verticalalignment='center')
            for i in range(self.station.nsta):
                if np.sum(self.station.used[i,:]):
                    ax0.plot(x[i],y[i],marker=(3,0,-self.station.azimuth[i]),color='green',zorder=101)
                else:
                    ax0.plot(x[i],y[i],marker=(3,0,-self.station.azimuth[i]),color='0.5',zorder=101)
                        
            for i in range(len(fm)):
                beach1 = beach(fm[i],xy=(i+0.5*i,0),width=width[i],show_iso=True)
                ax0.add_collection(beach1)
                ax0.text(i+0.5*i,0.55,fmtxt[i],horizontalalignment='center')
            
            for i, stat in enumerate(group):
                t = np.arange(0,self.station.npts[stat]*self.station.dt[stat],self.station.dt[stat])
                data = np.reshape(self.d[self.station.index1[stat,0]:self.station.index2[stat,-1]],
                                  (self.station.ncomp,self.station.npts[stat]))
                synt = np.reshape(Gm[self.station.index1[stat,0]:self.station.index2[stat,-1]],
                                  (self.station.ncomp,self.station.npts[stat]))
                ymin = np.min([data,synt])
                ymax = np.max([data,synt])
                for j in range(self.station.ncomp):
                    ax1[i,j].plot(t,data[j,:],color=datacol[stat,j])
                    ax1[i,j].plot(t,synt[j,:],color=syntcol[stat,j],dashes=[6,2])
                    ax1[i,j].set_ylim(ymin,ymax)
                    ax1[i,j].set_xlim(0,t[-1])
           
                ax1[i,0].set_yticks([ymin,0,ymax])
                ax1[i,0].set_yticklabels(['%.2e'%ymin, '0', '%.2e'%ymax])
                    
                ax1[i,0].text(0,ymax,
                              self.station.name[stat],
                              fontsize=10,fontweight='bold',verticalalignment='top')
                ax1[i,1].text(0,ymin,
                   r'($\mathit{r},\varphi$)=(%-.2f,%-.0f)'%(
                       self.station.distance[stat],self.station.azimuth[stat]),
                   fontsize=8,verticalalignment='bottom')
                ax1[i,2].text(0,ymin,'%d,%s'%(
                    self.station.shift[stat],stavr[stat]),fontsize=8,verticalalignment='bottom')
                
            f.savefig('figure%d.pdf'%(page))
            plt.close(f)
            
    def _read_green_library(self,precision=4):

        options = {'HERRMANN': ['_green_herrmann',('ss','ds','dd','ex')],
                   'TENSOR': ['_green_tensor',('xx','xy','xz','yy','yz','zz')]}
        greenlist = [ ''.join([c,suffix]) for suffix in options[self.header.greentype][1]
            for c in self.header.component ]
        
        G = [ None for _ in self.header.depth ]
        green = [ None for _ in self.header.depth ]
        for i, depth in enumerate(self.header.depth):            
            call_green_method = getattr(self,options[self.header.greentype][0])
            gg = call_green_method(depth,greenlist,precision)
            G[i] = gg[0]
            if self.header.correlate:
                green[i] = gg[1]
            
        self.G = G
        self.green = green
                
    def _green_tensor(self,depth,greenlist,precision):
        
        depth_str = '{:.{prec}f}'.format(depth,prec=precision)
        gg = dict.fromkeys(greenlist,None)
        G = np.zeros((np.sum(self.station.ncomp*self.station.npts),self.header.degree),dtype=np.float)
        v = [ getattr(self.station,k) for k in ('name','npts','index1','index2') ]
        for stat,npts,b,e in zip(*v):
            file = '%s/%s.%s'%(self.header.greendir,stat,depth_str)
            
            for suffix in greenlist:
                gg[suffix] = read('%s.%s'%(file,suffix))[0].data[0:npts]
                    
            # Read vertical then (if 3-C) horizontals
            # Construct the six basis functions (elemntary moment tensors)
            # Reference: Kikuchi and Kanamori,1991 (BSSA)
            for c,ii,jj in zip(self.header.component,b,e):
                G[ii:jj,0] = gg[c+'xy']
                G[ii:jj,1] = gg[c+'xx'] - gg[c+'yy']
                G[ii:jj,2] = gg[c+'yz']
                G[ii:jj,3] = gg[c+'xz']
                G[ii:jj,4] = -gg[c+'xx'] + gg[c+'zz']
                
                if self.header.degree == 6:
                    G[ii:jj,5] = gg[c+'xx'] + gg[c+'yy'] + gg[c+'zz']
                     
        return (G, None)
    
    def _green_herrmann(self,depth,greenlist,precision):
        """
        Load all station Green's functions
        """
        depth_str = '{:.{prec}f}'.format(depth,prec=precision)
        gg = dict.fromkeys(greenlist,None)
        green = np.zeros((np.sum(self.station.ncomp*self.station.npts),self.header.degree-2),dtype=np.float)
        G = np.zeros((np.sum(self.station.ncomp*self.station.npts),self.header.degree),dtype=np.float)
        v = [ getattr(self.station,k) for k in ('name','npts','azimuth','index1','index2') ]
        for stat,npts,azimuth,b,e in zip(*v):
            file = '%s/%s.%s'%(self.header.greendir,stat,depth_str)
            alpha = azimuth*(pi/180)
            for suffix in greenlist:
                if suffix == 'tdd' or suffix == 'tex':
                    gg[suffix] = np.zeros((npts,))
                else:
                    gg[suffix] = read('%s.%s'%(file,suffix))[0].data[0:npts]
            
            # Constrcut Green's function vector using equations 6, 7 and 8from Minson and Dreger, 2008 (GJI)
            # Some signs are flipped to match sign convention of basis Green's functions from RB Herrmann 2002,
            # Appendix B of Computer Programs in Seismology.
            # http://www.eas.slu.edu/eqc/eqccps.html

            # Vertical components
            G[b[0]:e[0],1] = gg['zss']*np.sin(2*alpha) # mxy
            G[b[0]:e[0],2] = gg['zds']*np.cos(alpha) # mxz
            G[b[0]:e[0],4] = gg['zds']*np.sin(alpha) # myz
            
            if self.header.degree == 5:
                G[b[0]:e[0],0] = 0.5*gg['zss']*np.cos(2*alpha) - 0.5*gg['zdd'] # mxx
                G[b[0]:e[0],3] = -0.5*gg['zss']*np.cos(2*alpha) - 0.5*gg['zdd'] # myy
            elif self.header.degree == 6:
                G[b[0]:e[0],0] = ( 0.5*gg['zss']*np.cos(2*alpha) - 0.166667*gg['zdd'] + 
                    0.33333*gg['zex'] ) # mxx
                G[b[0]:e[0],3] = ( -0.5*gg['zss']*np.cos(2*alpha) - 0.166667*gg['zdd'] + 
                    0.33333*gg['zex'] ) # myy
                G[b[0]:e[0],5] = 0.33333*gg['zdd'] + 0.33333*gg['zex'] # mzz
              
            # Read horizontals
            if self.header.component == 'zrt':
                G[b[2]:e[2],1] = -gg['tss']*np.cos(2*alpha) # mxy
                G[b[1]:e[1],1] = gg['rss']*np.sin(2*alpha)
                G[b[2]:e[2],2] = gg['tds']*np.sin(alpha) # mxz
                G[b[1]:e[1],2] = gg['rds']*np.cos(alpha)    
                G[b[2]:e[2],4] = -gg['tds']*np.cos(alpha) # myz
                G[b[1]:e[1],4] = gg['rds']*np.sin(alpha)
                
                G[b[2]:e[2],0] = 0.5*gg['tss']*np.sin(2*alpha) # mxx
                G[b[2]:e[2],3] = -0.5*gg['tss']*np.sin(2*alpha) # myy
                
                if self.header.degree == 5:
                    G[b[1]:e[1],0] = 0.5*gg['rss']*np.cos(2*alpha) - 0.5*gg['rdd'] # mxx
                    G[b[1]:e[1],3] = -0.5*gg['rss']*np.cos(2*alpha) - 0.5*gg['rdd'] # myy
                elif self.header.degree == 6:
                    G[b[1]:e[1],0] = 0.5*gg['rss']*np.cos(2*alpha) - 0.166667*gg['rdd'] + 0.33333*gg['rex']
                    G[b[1]:e[1],3] = -0.5*gg['rss']*np.cos(2*alpha) - 0.166667*gg['rdd'] + 0.33333*gg['rex']
                    G[b[1]:e[1],5] = 0.33333*gg['rdd'] + 0.33333*gg['rex'] # mzz
        
            # Read vertical then (if 3-C) horizontals
            for c,ii,jj in zip(self.header.component,b,e):
                green[ii:jj,0] = gg[c+'ss']
                green[ii:jj,1] = gg[c+'ds']
                green[ii:jj,2] = gg[c+'dd']
                if self.header.degree == 6:
                    green[ii:jj,3] = gg[c+'ex']
                                  
        return (G, green)
    
    def _read_data(self):
        # Read data
        data = [ None for _ in self.station.name ]
        for i in range(self.station.nsta):
            st = Stream()
            for c in self.header.component:
                st.append(read('%s/%s.%c.dat'%(self.header.datadir,self.station.name[i],c),format='SAC')[0])
            data[i] = st
        
        return data
        
    def _data2d(self,data,shift):
        # construct data vector according to time shifts and sample size
        d = np.zeros(np.sum(self.station.ncomp*self.station.npts),dtype=np.float)      
        v = [ getattr(self.station,k) for k in ('npts','index1','index2') ]
        for i, (nt,c1,c2) in enumerate(zip(*v)):
            for j in range(self.station.ncomp):
                d[c1[j]:c2[j]] =  data[i][j].data[shift[i]:shift[i]+nt]
        
        return d
        
    def _find_time_shift(self,data):
        # Cross-correlate data and Green's functions to estimate best time shift for each station
        shift = [ None for _ in self.header.depth ]
        #cc = np.zeros(shift.shape)
        G = getattr(self,{'TENSOR':'G','HERRMANN':'green'}[self.header.greentype])    
        v = [ getattr(self.station,k) for k in ('used','index1','index2','npts') ]

        _, ngf = G[0].shape
        for depth in range(len(self.header.depth)): # depth
            ts = np.copy(self.station.shift)
            for i,(used,b,e,nt) in enumerate(zip(*v)):
                if np.sum(used) > 0:
                    c = np.zeros((ngf,self.station.ncomp,data[i][0].stats.npts-nt+1))
                    c.fill(np.nan)
                    for j in range(self.station.ncomp):
                        for k in range(ngf):
                            c[k,j,:] = np.abs(xcorr(data[i][j].data,G[depth][b[j]:e[j],k]))
                c_average = np.average(c,axis=1)
                g1,g2 = np.unravel_index(c_average.argmax(), c_average.shape)
                ts[i] = g2
                #cc[i] = c_average[g1,g2]
            shift[depth] = ts
            
        self.shift = shift

    def _calculate_weights(self):
        # Determin which weight function to used
        options = {0:'_weight_equal',1:'_weight_inverse_distance',2:'_weight_inverse_variance'}
        call_method = getattr(self,options[self.header.weight])
        call_method()        

    def _weight_equal(self):
        # No weights applied
        stawei = np.ma.masked_where(np.sum(self.station.used,axis=1)==0,np.ones((self.station.nsta),dtype=np.float))
        stawei.filled()
        self.stawei = [stawei]
        self.w = [np.repeat(
            (self.station.used.T*self.stawei).T,np.repeat(self.station.npts,self.station.ncomp))]
    
    def _weight_inverse_distance(self):
        # Inverse distance
        dist = np.ma.masked_where(np.sum(self.station.used,axis=1)==0,self.station.distance)
        self.stawei = [dist/np.min(dist)]
        self.w = [np.repeat(
            (self.station.used.T*self.stawei).T,np.repeat(self.station.npts,self.station.ncomp))]
        
    def _weight_inverse_variance(self):
        # Inverse data variance
        stawei = [ None for _ in self.header.depth ]
        w = [ None for _ in self.header.depth ]
        for i in range(len(self.header.depth)):
            wei = np.ma.masked_array(np.ones((self.station.nsta),dtype=np.float))
            dd = np.ma.masked_where(
                 np.repeat(self.station.used,np.repeat(self.station.npts,self.station.ncomp))==0,
                    self.d[i])
            for j, (b, e) in enumerate(zip(self.station.index1[:,0],self.station.index2[:,-1])):
                wei[j] = 1/np.sum(dd[b:e]**2)
            
            stawei[i] = wei/np.sum(wei)
            w[i] = np.repeat(
                (self.station.used.T*stawei[i]).T,np.repeat(self.station.npts,self.station.ncomp))

        self.stawei = stawei
        self.w = w
             
    def _run_inversion(self,cores=2):
        """
        Parallel processing
        Run :func: invert in parallel, and decomposition is applied to all solutions
        """
        p = mp.Pool(processes=cores)
        if self.header.weight == 2:
            res = [p.apply_async(invert, args=(
                self.d[i],self.G[i],self.w[i],
                self.header.greentype,
                self.header.degree,
                self.header.plot,
                self.station.nsta,
                self.station.used,
                self.station.index1,
                self.station.index2)
                ) for i, _ in enumerate(self.header.depth)]
        else:
            res = [p.apply_async(invert, args=(
                self.d[i],self.G[i],self.w[0],
                self.header.greentype,
                self.header.degree,
                self.header.plot,
                self.station.nsta,
                self.station.used,
                self.station.index1,
                self.station.index2)
                ) for i, _ in enumerate(self.header.depth)]
            
        self.solutions = [ r.get() for r in res ]
        
    def _write_solutions2list(self,file='mtinv.out'):
        """
        Write all solutions to a text file
        columns:
        depth, degree,mw,mo,mxx---myz,pdc,pclvd,piso,str,dip,rake,gamma,delta,VR
        """
        with open(file,'w') as f:
            f.write('depth degree Mw Mo Mxx Myy Mzz Mxy Mxz Myz pDC pCLVD pISO Str1 Rake1 Dip1 Str2 Rake2 Dip2 Gamma Delta VR\n')
            order = ( i for i in range(len(self.header.depth)) )
            order = ( self.best, *order )
            for i in order:
                f.write('{:6.3f} {:1d} '.format(self.header.depth[i],self.header.degree))
                f.write('{:4.2f} {:10.3e} '.format(self.solutions[i]['mw'],self.solutions[i]['mo']))
                f.write(' '.join('{:10.3e} '.format(elm) for elm in self.solutions[0]['M']))
                f.write('{:3.0f} {:3.0f} {:3.0f}'.format(self.solutions[i]['pdc'],
                                                         self.solutions[i]['pclvd'],
                                                         self.solutions[i]['piso']))
                f.write(' '.join('{:4.0f} '.format(deg) for deg in self.solutions[i]['fps'][0]))
                f.write(' '.join('{:3.0f} '.format(deg) for deg in self.solutions[i]['fps'][1]))
                f.write(' '.join('{:7.2f} '.format(deg) for deg in self.solutions[i]['lune']))
                f.write('{:6.2f}\n'.format(self.solutions[i]['VR']))
        
    def _write_best2text(self):
        # Write best solution to a text file
        solution = self.solutions[self.best]
        stawei = [ '%.3f'%val if val else '     ' for val in self.stawei ]
        stavr = [ '%.0f'%val if val else '' for val in solution['staVR'] ]
        
        with open('max.mtinv.out','w') as f:    
            f.write('%d-Degree Moment Tensor Inversion\n'%self.header.degree)
            f.write('Depth = %6.3f (km)\n'%(self.header.depth[self.best]))
            f.write('Mo = %.3e (dyne-cm)\n'%solution['mo'])
            f.write('Mw = %.2f\n'%solution['mw'])
            f.write('Percent DC   = %3d\n'%solution['pdc'])
            f.write('Percent CLVD = %3d\n'%solution['pclvd'])
            f.write('Percent ISO  = %3d\n'%solution['piso'])
            f.write('Fault Plane 1: Strike=%-3.0f Dip=%-2.0f Rake=%-3.0f\n'%solution['fps'][0])
            f.write('Fault Plane 2: Strike=%-3.0f Dip=%-2.0f Rake=%-3.0f\n'%solution['fps'][1])
            f.write('Percent Variance Reduction = %.2f %%\n'%solution['VR'])

            f.write('\nMoment Tensor Elements: Cartesian Coordinates (1e20 dyne-cm)\n')
            f.write('Mxx     Myy     Mzz     Mxy     Mxz     Myz\n')
            f.write(''.join('%-10.3e '%(elm) for elm in solution['M']))

            f.write('\n\nEigenvalues: ')
            f.write(' '.join('%.3e'%i for i in solution['eigenvalues']))
            f.write('\nLune Coordinates: ')
            f.write(' '.join('%-6.2f'%i for i in solution['lune']))
            
            f.write('\n\nStation Information\n')
            for i in range(self.station.nsta):
                f.write('Station: %s R=%-.1f AZI=%-5.1f '%(self.station.name[i],
                                                           self.station.distance[i],
                                                           self.station.azimuth[i]))
                f.write('W=%-5s '%stawei[i])
                f.write('Shift=%d '%self.station.shift[i])
                f.write('VR=%s\n'%stavr[i])
            
if __name__ == "__main__":

    def call_inversion(filename):
        m = loadfile(filename)
        m.run()
        m.write()

    if len(sys.argv) < 2:
        filename = 'mtinv.in'
    else:
        filename = sys.argv[1]

    call_inversion(filename)

