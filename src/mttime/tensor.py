# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea@llnl.gov)
"""
Routines for handling inversion results
"""

import numpy as np


def get_m_in_basis(m, in_basis, out_basis):
    """
    Convert moment tensor between coordinate systems

    A function that returns the moment tensor in a new coordinate system.
    See supported coordinate systems below.

    :param m: moment tensor elements (M11, M22, M33, M12, M13, M23) for a given
        coordinate system (1, 2, 3).
    :type m: :class:`~numpy.ndarray` or list
    :param in_basis: input moment tensor coordinate system.
    :type in_basis: str
    :param out_basis: output moment tensor coordinate system.
    :type out_basis: str
    :return: moment tensor in a new coordinate system.
    :rtype: :class:`~numpy.ndarray`

    .. rubric:: Supported coordinate systems:

    .. cssclass: table-striped

    ================   ==================   ==========================
    ``basis``          vectors              reference
    ================   ==================   ==========================
    "NED"              north, east, down    Jost and Herrmann, 1989
    "XYZ"              north, east, down    Aki and Richard, 1980
    "USE"              up, south, east      Larson et al., 2010
    "RTP"              r, theta, phi        Harvard/Global CMT convention
    ================   ==================   ==========================
    """
    # Mrr = Mzz, Mtt = Mxx, Mpp = Myy, Mrt = Mxz, Mrp = -Myz, Mtp = -Mxy
    # name : component                  NED sign and indices
    # NED  : NN, EE, DD, NE, ND, ED --> [0, 1, 2, 3, 4, 5]
    # XYZ  : XX, YY, ZZ, XY, XZ, YZ --> [0, 1, 2, 3, 4, 5]
    # RTP  : RR, TT, PP, RT, RP, TP --> [1, 2, 0, -5, 3, -4]
    # USE  : UU, SS, EE, US, UE, SE --> [1, 2, 0, -5, 3, -4]

    allowed_bases = ["NED", "XYZ", "RTP", "USE"]
    if in_basis not in allowed_bases:
        msg = "moment tensor in {:s} coordinates not supported."
        raise NotImplementedError(msg.format(in_basis))

    if out_basis not in allowed_bases:
        msg = "moment tensor in {:s} coordinates not supported."
        raise NotImplementedError(msg.format(out_basis))

    if in_basis == out_basis:
        return m
    else:
        if in_basis == "USE" or in_basis == "RTP":
            signs = [1, 1, 1, -1, 1, -1]
            indices = [1, 2, 0, 5, 3, 4]

        elif in_basis == "NED" or in_basis == "XYZ":
            signs = [1, 1, 1, 1, -1, -1]
            indices = [2, 0, 1, 4, 5, 3]
        else:
            msg = "Unable to convert moment tensor."
            raise NotImplementedError(msg)

        return np.array([sign * m[i] for sign, i in zip(signs, indices)], dtype=np.float64)


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
    Moment tensor elements will be stored in XYZ coordinates. Use
    :meth:`~mttime.tensor.Tensor.get_tensor_elements` to convert between
    coordinate systems.

    :param m: six independent moment tensor elements
        (M11, M22, M33, M12, M13, M23) in dyne-cm.
    :type m: :class:`~numpy.ndarray`
    :param basis: moment tensor coordinate system. Default is ``"XYZ"``.
    :type basis: str
    :param depth: source depth.
    :type depth: float
    :param inversion_type: ``Deviatoric`` or ``Full``.
    :type inversion_type: str
    :param component: waveform components.
    :type component: list of str
    :param station_table: station table. Required header names are: station, distance, azimuth,
        ts, dt, weights, VR, [ZRTNE] (one column for each component in ``component``), longitude,
        and latitude.
    :type station_table: :class:`~pandas.core.frame.DataFrame`
    :param total_VR: total variance reduction.
    :type total_VR: float
    :param dd: observed data stored as a single vector, required for plotting.
    :type dd: :class:`~numpy.ndarray`
    :param ss: synthetic seismograms stored as a single vector, required for plotting.
    :type ss: :class:`~numpy.ndarray`
    """
    def __init__(self, m, basis="XYZ", **kwargs):
        self._input_basis = basis
        self._m = m
        self._parse_inversion_info(**kwargs)

    @property
    def m(self):
        """
        Moment tensor elements

        MXX, MYY, MZZ, MXY, MXZ, MYZ in dyne-cm.
        """
        return self._m

    @m.setter
    def m(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = np.array(value, dtype=np.float64)
        else:
            msg = "Not a valid moment tensor."
            raise ValueError(msg)

        self._m = get_m_in_basis(value, self._input_basis, "XYZ")

    def get_tensor_elements(self, basis="XYZ"):
        """
        Retrieve the moment tensor elements

        Function that returns the moment tensor in the specified system
        coordinates.

        :param basis: system coordinates, default is ``XYZ``. Refer to
            :func:`~mttime.tensor.get_m_in_basis` for supported systems.
        :type basis: str
        :return: a dictionary of moment tensor elements.
        :rtype: dict

        """
        system_mapping = dict(XYZ=["XX", "YY", "ZZ", "XY", "XZ", "YZ"],
                              NED=["NN", "EE", "DD", "NE", "ND", "ED"],
                              RTP=["RR", "TT", "PP", "RT", "RP", "TP"],
                              USE=["UU", "SS", "EE", "US", "UE", "SE"],
        )
        m = get_m_in_basis(self._m, "XYZ", basis)
        out = dict()
        for key, value in zip(system_mapping[basis], m):
            out["M%s"%key] = value

        return out

    def decompose(self):
        """
        Moment tensor decomposition

        A function that decomposes the moment tensor from attribute ``m``.
        Moment is in dyne-cm.

        .. rubric:: _`Attributes`

        ``iso``: :class:`~numpy.ndarray`
            isotropic moment tensor elements.
        ``dev``: :class:`~numpy.ndarray`
            deviatoric moment tensor elements.
        ``dc``: :class:`~numpy.ndarray`
            double couple moment tensor elements.
        ``clvd``: :class:`~numpy.ndarray`
            CLVD moment tensor elements.
        ``eigenvalues``: :class:`~numpy.ndarray`
            eigenvalues.
        ``mo``: float
            total scalar seismic moment in Bowers and Hudson (1999) convention.
        ``mw``: float
            moment magnitude.
        ``mw_dev``: float
            deviatoric moment magnitude.
        ``miso``: float
            isotropic moment.
        ``mdc``: float
            double-couple moment.
        ``mclvd``: float
            CLVD moment.
        ``pdc``: float
            percentage of double-couple component.
        ``pclvd``: float
            percentage of CLVD component.
        ``piso``: float
            percentage of isotropic component.
        ``fps``: list of (float, float, float)
            strike, dip, and rake of the two fault planes.
        ``lune``: (float, float)
            gamma and delta in lune source-type space.
        """
        m = self._m
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
        self.iso = np.array(
            [M_iso[0, 0], M_iso[1, 1], M_iso[2, 2], M_iso[0, 1], M_iso[0, 2], M_iso[1, 2]]
        )
        self.dev = np.array(
            [M_dev[0, 0], M_dev[1, 1], M_dev[2, 2], M_dev[0, 1], M_dev[0, 2], M_dev[1, 2]]
        )
        self.dc = np.array(
            [M_dc[0, 0], M_dc[1, 1], M_dc[2, 2], M_dc[0, 1], M_dc[0, 2], M_dc[1, 2]]
        )
        self.clvd = np.array(
            [M_clvd[0, 0], M_clvd[1, 1], M_clvd[2, 2], M_clvd[0, 1], M_clvd[0, 2], M_clvd[1, 2]]
        )
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

    def write(self):
        """
        Write detailed solution to file

        A function that saves the detailed moment tensor solution to a file named
        `d[depth].mtinv.out`.
        """
        filename = "d%07.4f.mtinv.out" %self.depth
        cmt = get_m_in_basis(self._m, "XYZ", "RTP")
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
                "Harvard/CMT convention\n"
                "Mrr        Mtt        Mpp        Mrt        Mrp        Mtp\n"
                "{cmt[0]:<10.3e} {cmt[1]:<10.3e} {cmt[2]:<10.3e} {cmt[3]:<10.3e} {cmt[4]:<10.3e} {cmt[5]:<10.3e}\n\n"
                "Eigenvalues: {eigenvalues[0]:.3e} {eigenvalues[1]:.3e} {eigenvalues[2]:.3e}\n"
                "Lune Coordinates: {lune[0]:<6.2f} {lune[1]:<6.2f}\n"
                "Station Information\n"
               )
        out = out.format(type=self.inversion_type,
                         depth=self.depth,
                         mo=self.mo,
                         mw=self.mw,
                         pdc=self.pdc,
                         pclvd=self.pclvd,
                         piso=self.piso,
                         fps=self.fps,
                         VR=self.total_VR,
                         m=self._m,
                         cmt=cmt,
                         eigenvalues=self.eigenvalues,
                         lune=self.lune
                        )

        # Update string print format
        self.station_table["VR"] = self.station_table["VR"].map("{:.2f}".format)
        self.station_table["weights"] = self.station_table["weights"].map("{:.4f}".format)

        with open(filename, "w") as f:
            f.write(out)
            f.write("%s\n" % (self.station_table.to_string(index=False)))

    def _parse_inversion_info(self, **kwargs):
        self.depth = kwargs.get("depth")
        self.inversion_type = kwargs.get("inversion_type")
        self.components = kwargs.get("components")
        self.total_VR = kwargs.get("total_VR")

        self.station_table = kwargs.get("station_table").copy(deep=True)
        self.station_table["VR"] = kwargs.get("station_VR")

        self._data = kwargs.get("dd")
        self._synthetics = kwargs.get("ss")

    def _get_summary(self):
        """
        Returns a brief summary of the inversion
        """
        out = (
            "{inversion_type:s} Moment Tensor Inversion\n"
            "Depth = {depth:.4f} km\n"
            "Mw = {mw:.2f}\n"
            "Percent DC/CLVD/ISO = {pdc:.0f}/{pclvd:.0f}/{piso:.0f}\n"
            "VR = {VR:.2f}%\n"
        )
        out = out.format(
            inversion_type=self.inversion_type,
            depth=self.depth,
            mw=self.mw,
            pdc=self.pdc,
            pclvd=self.pclvd,
            piso=self.piso,
            VR=self.total_VR
        )

        return out

    def __str__(self):
        keys = ["inversion_type", "depth", "mw", "total_VR"]
        #for key in keys:
        #    if key in self.__dict__.keys():

        ret = ("Mxx={m[0]:>10.3e} Myy={m[1]:>10.3e} Mzz={m[2]:>10.3e}\n"
               "Mxy={m[3]:>10.3e} Mxz={m[4]:>10.3e} Myz={m[5]:>10.3e}\n"
               )

        return ret.format(m=self._m)
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

