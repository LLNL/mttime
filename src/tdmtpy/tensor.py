# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
Routines for handling inversion results
"""

import numpy as np
from obspy.core.util import AttribDict

import matplotlib.pyplot as plt
from .image import new_page, beach


def find_strike_rake_dip(u,n):
    """
    Compute strike,rake and dip

    A function that returns the strike, rake and dip of given slip
    and fault normal vectors.

    :param u: slip vector.
    :type u: :class:`~numpy.ndarray`
    :param n: fault normal vector.
    :type n: :class:`~numpy.ndarray`
    :return: strike, rake and dip of a fault plane.
    :rtype: (float, float, float)
    """
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
    """
    Compute direction of slip and orientation of fault planes

    Function that returns slip direction and orientation of fault plane
    and auxiliary plane.

    :param M: moment tensor in matrix form.
    :type M: :class:`~numpy.ndarray`
    :param M_dc: double-couple moment tensor in matrix form.
    :type M_dc: :class:`~numpy.ndarray`
    :return: direction of slip and orientation of the fault plane
        and auxiliary plane.
    :rtype: list((float, float, float), (float, float, float))
    """
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
    Calculate source-type parameters on a lune

    A function that calculates the source-type parameters gamma and delta
    based on the formulation of Tape and Tape, (2012).

    :param lam: eigenvalues in descending order.
    :type lam: :class:`~numpy.ndarray`
    :return: lune coordinates gamma and delta.
    :rtype: (float, float)
    """
    lambda_mag = np.sqrt(lam[0]**2 + lam[1]**2 + lam[2]**2)
        
    if np.sum(lam) != 0:
        bdot = np.sum(lam)/(np.sqrt(3)*lambda_mag)
        if bdot > 1:
            bdot = 1
        elif bdot < -1:
            bdot = -1
        delta = 90 - np.arccos(bdot)*180/np.pi
    else:
        delta = 0.
            
    if lam[0] == lam[2]:
        gamma = 0
    else:
        gamma = np.arctan((-lam[0] + 2*lam[1] - lam[2]) / (np.sqrt(3)*(lam[0] - lam[2])))*180/np.pi
        
    return (gamma,delta)


class Tensor(object):
    """
    Object containing a single six-element moment tensor

    A container for all things related to the moment tensor ``m``.
    Optional ``**kwargs`` are used to write and plot inversion results.

    :param m: six independent moment tensor elements in Aki and Richards (2002) convention:
        mxx, myy, mzz, mxy, mxz, myz. Units are in dyne-cm.
    :type m: :class:`~numpy.ndarray`
    :param depth: source depth.
    :type depth: float
    :param ts: data shift in the number of time points.
    :type ts: int
    :param weights: station weights.
    :type weights: :class:`~numpy.ndarray`
    :param station_VR: variance reduction at each station.
    :type station_VR: :class:`~numpy.ndarray`
    :param total_VR: total variance reduction.
    :type total_VR: float
    :param dd: observed data stored as a single vector, required for plotting.
    :type dd: :class:`~numpy.ndarray`
    :param ss: synthetic seismograms stored as a single vector, required for plotting.
    :type ss: :class:`~numpy.ndarray`

    :var inverted: inversion related parameters from ``**kwargs``.
    :vartype inverted: dict
    :var cmt: moment tensor elements in GCMT format: mrr, mtt, mpp, mrt, mrp, mtp.
        Units are in N-m.
    :vartype cmt: :class:`~numpy.ndarray`
    :var iso: isotropic moment tensor elements.
    :vartype iso: :class:`~numpy.ndarray`
    :var dev: deviatoric moment tensor elements.
    :vartype dev: :class:`~numpy.ndarray`
    :var dc: double couple moment tensor elements.
    :vartype dc: :class:`~numpy.ndarray`
    :var clvd: CLVD moment tensor elements.
    :vartype clvd: :class:`~numpy.ndarray`
    :var eigenvalues: eigenvalues.
    :vartype eigenvalues: :class:`~numpy.ndarray`
    :var mo: total scalar seismic moment in Bowers and Hudson (1999) convention.
        Unit is in dyne-cm.
    :vartype mo: float
    :var mw: moment magnitude in dyne-cm.
    :vartype mw: float
    :var mw_dev: deviatoric moment magnitude
    :vartype mw_dev: float
    :var miso: isotropic moment in dyne-cm.
    :vartype miso: float
    :var mdc: double-couple moment in dyne-cm.
    :vartype mdc: float
    :var mclvd: CLVD moment in dyne-cm.
    :vartype mclvd: float
    :var pdc: percentage of double-couple component.
    :vartype pdc: float
    :var pclvd: percentage of CLVD component.
    :vartype pclvd: float
    :var piso: percentage of isotropic component.
    :vartype piso: float
    :var fps: strike, dip, and rake of the two fault planes.
    :vartype fps: list((float, float, float), (float, float, float))
    :var lune: gamma and delta in lune source-type space.
    :vartype lune: (float, float)
    """
    def __init__(self,m=None,**kwargs):
        self._set_m(m)
        # Optional keywords default to None
        inverted = AttribDict()

        keys = ["depth","components","ts","weights","station_VR","total_VR","dd","ss"]
        # Update given key by their given value
        for key, value in kwargs.items():
            if key in keys:
                inverted[key] = value
        self.inverted = inverted

    def _set_m(self,m):
        if m is None:
            msg = "Please provide a moment tensor."
            raise ValueError(msg)

        self.m = m

    def cartesian_to_spherical(self):
        """
        Conversion from cartesian coordinates to spherical coordinates

        A function that converts moment tensor elements from attribute ``m`` to GCMT format
        and assigns the vector of tensor elements (mrr, mtt, mpp, mrt, mrp, mtp)
        to attribute ``cmt``.

        Conversion is defined by:

        Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy,

        where x, y, z equal to north, east, down (Aki and Richards, 2002).
        """
        cmt = np.zeros(6,)
        cmt[0] = self.m[2]
        cmt[1] = self.m[0]
        cmt[2] = self.m[1]
        cmt[3] = self.m[4]
        cmt[4] = -1*self.m[5]
        cmt[5] = -1*self.m[3]

        self.cmt = cmt/1e7 # convert to Newton-meters

    def decompose(self):
        """
        Moment tensor decomposition

        A function that decomposes the moment tensor from attribute ``m`` and assigns the result
        to various other attributes. Refer to :class:`~tdmtpy.tensor.Tensor` for the full
        list of variables after decomposition.
        """
        m = self.m
        M = np.array([[m[0], m[3], m[4]],
                      [m[3], m[1], m[5]],
                      [m[4], m[5], m[2]]]
                     )

        # Isotropic part:
        M_iso = np.diag(np.array([1. / 3 * np.trace(M),
                                  1. / 3 * np.trace(M),
                                  1. / 3 * np.trace(M)]
                                )
                        )
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
#       m2 = dev_eigval[1]
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
        
        # Bowers and Hudson's definition of seismic moment
        mo = abs(miso)+abs(m3) # iso moment + dev moment
        mw = (np.log10(mo)-16.05)*2/3 # dyne-cm
        mw_dev = 2*np.log10(mo)/3-10.73

        # Calculate percentage
        piso = abs(miso)/mo*100
        pdc = mdc/mo*100
        pclvd = mclvd/mo*100
        
        # DC Fault planes
        fps =  find_fault_planes(M,M_dc)
    
        # Find gamma and delta, Tape&Tape lune parameters
        lune = eigen2lune(eigenvalues)
    
        #self.M = np.array([M[0,0],M[1,1],M[2,2],M[0,1],M[0,2],M[1,2]])
        self.iso = np.array([M_iso[0,0],M_iso[1,1],M_iso[2,2],M_iso[0,1],M_iso[0,2],M_iso[1,2]])
        self.dev = np.array([M_dev[0,0],M_dev[1,1],M_dev[2,2],M_dev[0,1],M_dev[0,2],M_dev[1,2]])
        self.dc = np.array([M_dc[0,0],M_dc[1,1],M_dc[2,2],M_dc[0,1],M_dc[0,2],M_dc[1,2]])
        self.clvd = np.array([M_clvd[0,0],M_clvd[1,1],M_clvd[2,2],M_clvd[0,1],M_clvd[0,2],M_clvd[1,2]])
        self.eigenvalues = eigenvalues
        self.mo = mo
        self.mw = mw
        self.mw_dev = mw_dev
        self.miso = miso
        self.mdc = mdc
        self.mclvd = mclvd
        self.pdc = pdc
        self.pclvd = pclvd
        self.piso = piso
        self.fps = fps
        self.lune = lune

    def write(self,inv_type,df):
        """
        Write detailed solution file

        A function that saves the detailed moment tensor solution to a file named
        `d[depth].mtinv.out`, where depth is the inverted source depth in attribute
        ``inverted.depth``.

        :param inv_type: type of moment tensor inversion, ``deviatoric`` or ``full``.
        :type inv_type: str
        :param df: updated station table from attribute ``config.df`` in
            :class:`~tdmtpy.inversion.Inversion`.
        :type df: :class:`~pandas.DataFrame`
        """
        filename = "d%07.4f.mtinv.out" %self.inverted.depth
        self.cartesian_to_spherical()
        out = ("{type:s} Moment Tensor Inversion\n"
               "Depth = {depth:7.4f} (km)\n"
               "Mo = {mo:.3e} (dyne-cm)\n"
               "Mw = {mw:.2f}\n"
               "Percent DC   = {pdc:3.0f}\n"
               "Percent CLVD = {pclvd:3.0f}\n"
               "Percent ISO  = {piso:3.0f}\n"
               "Fault Plane 1: Strike={fps[0][0]:<3.0f} Dip={fps[0][1]:<2.0f} Rake={fps[0][2]:<3.0f}\n"
               "Fault Plane 2: Strike={fps[1][0]:<3.0f} Dip={fps[1][1]:<2.0f} Rake={fps[1][2]:<3.0f}\n"
               "Percent Variance Reduction = {VR:.2f}\n"
                "\n"
                "Moment Tensor Elements: Aki and Richards Cartesian Coordinates\n"
                "Mxx        Myy        Mzz        Mxy        Mxz        Myz\n"
                "{m[0]:<10.3e} {m[1]:<10.3e} {m[2]:<10.3e} {m[3]:<10.3e} {m[4]:<10.3e} {m[5]:<10.3e}\n\n"
                "GCMT convention (N-m)\n"
                "Mrr        Mtt        Mpp        Mrt        Mrp        Mtp\n"
                "{cmt[0]:<10.3e} {cmt[1]:<10.3e} {cmt[2]:<10.3e} {cmt[3]:<10.3e} {cmt[4]:<10.3e} {cmt[5]:<10.3e}\n\n"
                "Eigenvalues: {eigenvalues[0]:.3e} {eigenvalues[1]:.3e} {eigenvalues[2]:.3e}\n"
                "Lune Coordinates: {lune[0]:<6.2f} {lune[1]:<6.2f}\n"
                "Station Information\n"
               )
        out = out.format(type=inv_type,
                         depth=self.inverted.depth,
                         mo=self.mo,
                         mw=self.mw,
                         pdc=self.pdc,
                         pclvd=self.pclvd,
                         piso=self.piso,
                         fps=self.fps,
                         VR=self.inverted.total_VR,
                         m=self.m,
                         cmt=self.cmt,
                         eigenvalues=self.eigenvalues,
                         lune=self.lune
                        )

        # Update station table
        df = df.copy()
        df["ts"] = value = self.inverted.ts
        df["VR"] = self.inverted.station_VR
        df["VR"] = df["VR"].map("{:.2f}".format)
        df["weights"] = self.inverted.weights
        df["weights"] = df["weights"].map("{:.4f}".format)

        # columns = ["station","distance","azimuth","ts","npts","dt",""]
        with open(filename, "w") as f:
            f.write(out)
            f.write("%s\n" % (df.to_string(index=False)))

    def _beach_waveforms_3c(self,inv_type,components,df,show,format):
        """
        Plot waveform fits and focal mechanisms
        """

        # Turn interactive plotting off
        plt.ioff()  # only display plots when called, save figure without displaying in ipython

        # Max. 10 stations per figure
        nsta = len(df.index)
        nrows, ncols = 10, 3  # row=station,col=component
        a = nsta / nrows
        nPages = int(a) + ((int(a) - a) != 0)
        lst = list(range(0, nsta, nrows))
        lst.append(nsta)
        pages = (range(lst[i], lst[i + 1]) for i in range(nPages))

        # Brief summary of inversion
        annot = ("{inv_type:s} Moment Tensor Inversion\n"
                 "Depth = {depth:.4f} km\n"
                 "Mw = {mw:.2f}\n"
                 "Percent DC/CLVD/ISO = {pdc:.0f}/{pclvd:.0f}/{piso:.0f}\n"
                 "VR = {VR:.2f}%\n"
                 )
        annot = annot.format(inv_type=inv_type,
                             depth=self.inverted.depth,
                             mw=self.mw,
                             pdc=self.pdc,
                             pclvd=self.pclvd,
                             piso=self.piso,
                             VR=self.inverted.total_VR
                             )

        # Decompositions to plot
        if inv_type == "Deviatoric":
            fm_title = ["Deviatoric", "DC", "CLVD"]
            fm = [self.m, self.fps[0], self.clvd]
            fm_width = [1,0.01*self.pdc,0.01*self.pclvd]
        elif inv_type == "Full":
            fm_title = ['Full', 'ISO', 'Deviatoric']
            fm = (self.m, self.iso, self.dev)
            fm_width = [1, 0.01*self.piso, 0.01*(self.pdc+self.pclvd)]
        fm_sign = 1 - 0.25 * fm_width[1], 2.5 - 0.5 * fm_width[2]

        # Station location around beach ball
        x = 0.55 * np.sin(df.azimuth * np.pi / 180).values
        y = 0.55 * np.cos(df.azimuth * np.pi / 180).values
        tri_color = np.array(np.repeat("0.5",nsta),dtype="<U5")
        tri_color[df[components].sum(axis=1) > 0] = "green"

        # Waveform line colors
        syntcol = np.empty(df[components].shape, dtype='<U5')
        syntcol[df[components] == 1] = 'green'
        syntcol[df[components] == 0] = '0.5'
        datacol = np.empty(df[components].shape, dtype='<U5')
        datacol[df[components] == 1] = 'black'
        datacol[df[components] == 0] = '0.5'

        for page, group in enumerate(pages):
            f, ax0, ax1 = new_page(len(group), nrows+1, ncols, annot=annot,title=self.inverted.components)
            # Plot beach balls
            for i in range(len(fm)):
                beach1 = beach(fm[i], xy=(i + 0.5 * i, 0), width=fm_width[i], show_iso=True)
                ax0.add_collection(beach1)
                ax0.text(i + 0.5 * i, 0.55, fm_title[i], horizontalalignment='center')
            ax0.text(fm_sign[0], 0, '=', horizontalalignment='center', verticalalignment='center')
            ax0.text(fm_sign[1], 0, '+', horizontalalignment='center', verticalalignment='center')
            # Plot stations around beach ball
            for xi,yi,azi,col in zip(x,y,df.azimuth,tri_color):
                ax0.plot(xi, yi, marker=(3, 0, -1*azi), color=col, zorder=101, markersize=5)
            # Plot waveforms
            for i, stat in enumerate(group):
                t = np.arange(0, df.npts[stat] * df.dt[stat], df.dt[stat])
                data = self.inverted.dd[stat]
                synt = self.inverted.ss[stat]
                ymin = np.min([data, synt])
                ymax = np.max([data, synt])
                for j in range(len(components)):
                    ax1[i, j].plot(t, data[:, j], color=datacol[stat, j], clip_on=False)
                    ax1[i, j].plot(t, synt[:, j], color=syntcol[stat, j], dashes=[6, 2], clip_on=False)
                    ax1[i, j].set_ylim(ymin, ymax)
                    ax1[i, j].set_xlim(0, t[-1])
                # Set ticks and labels
                ax1[i, 0].set_yticks([ymin, 0, ymax])
                ax1[i, 0].set_yticklabels(['%.2e' % ymin, '0', '%.2e' % ymax])
                # Station name, distance and azimuth
                if df.distance[stat] > 10:
                    dist = ("{0:.0f}")
                else:
                    dist = ("{0:.4f}")
                dist = dist.format(df.distance[stat])
                label = '\n'.join([df.station[stat],
                                  r'$\Delta,\theta$=%s,%-.0f' % (dist, df.azimuth[stat])
                                  ])
                ax1[i, 0].text(0,ymax, label, verticalalignment="bottom")
                # Sample shift and VR
                ax1[i, 1].text(t[-1], ymax,
                               'ts,VR=%d,%.0f'%(self.inverted.ts[stat], self.inverted.station_VR[stat]),
                               horizontalalignment="right",verticalalignment="bottom")
            # Label last row only
            for column in range(3):
                ax1[i, column].set_xlabel('Time [s]')

            outfile = "bbwaves.d%07.4f.%02d.%s"%(self.inverted.depth,page,format)
            f.savefig(outfile,format=format,bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(f)

    def __str__(self):
        ret = ("Mxx={m[0]:>10.3e} Myy={m[1]:>10.3e} Mzz={m[2]:>10.3e}\n"
               "Mxy={m[3]:>10.3e} Mxz={m[4]:>10.3e} Myz={m[5]:>10.3e}\n"
               )

        return ret.format(m=self.m)
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
