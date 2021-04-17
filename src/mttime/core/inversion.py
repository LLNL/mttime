# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea@llnl.gov)
"""
Time domain moment tensor inverse routines

.. rubric:: Example

.. code-block:: python

    >>> config = mttime.Configure(path_to_file="examples/synthetic/mtinv.in")
    >>> tdmt = mttime.Inversion(config=config)
    >>> print(tdmt)
         event: {'datetime': '2019-07-16T20:10:31.473', 'longitude': -121.757, 'latitude': 37.8187}
         depth: [10.0, 20.0]
         green: herrmann
    components: ['Z', 'R', 'T']
        degree: 5
        weight: distance
          plot: False
     correlate: False
    | STATION TABLE |
        station  distance  azimuth  ts  npts   dt  Z  R  T  longitude  latitude filter  nc  np  lcrn  hcrn model
     BK.FARB.00    110.00   263.00  30   100 1.00  0  0  0    -123.00     37.70     bp   2   2  0.05  0.10  gil7
      BK.SAO.00    120.00   167.00  30   150 1.00  0  0  1    -121.45     36.76     bp   2   2  0.05  0.10  gil7
      BK.CMB.00    123.00    78.00  30   150 1.00  0  0  0    -120.39     38.03     bp   2   2  0.05  0.10  gil7
     BK.MNRC.00    132.00   333.00  30   150 1.00  1  1  1    -122.44     38.88     bp   2   2  0.05  0.10  gil7
    | PREFERRED SOLUTION |
    None
    >>> tdmt.invert()
    Deviatoric Moment Tensor Inversion
    Depth = 10.0000 km
    Mw = 3.97
    Percent DC/CLVD/ISO = 100/0/0
    VR = 100.00%%

    Deviatoric Moment Tensor Inversion
    Depth = 20.0000 km
    Mw = 4.25
    Percent DC/CLVD/ISO = 62/38/0
    VR = 87.05%%
"""

from copy import deepcopy
import numpy as np
from obspy.core import read, Stream

from mttime.core import utils
from mttime.core.tensor import Tensor


class Inversion(object):
    """
    A container for a single inverse routine

    Object contains the inversion parameters, data and synthetic files, and moment tensor
    solutions.

    :param config: object containing the necessary parameters for setting up the inverse routine
    :type config: :class:`~mttime.core.configure.Configure`

    :var streams: processed waveform data.
    :vartype streams: a list of :class:`~obspy.core.stream.Stream`
    :var moment_tensors: moment tensor solutions.
    :vartype moment_tensors: a list of :class:`~mttime.core.tensor.Tensor`
    :var preferred_tensor_id: index to the preferred moment tensor solution
        (maximum variance reduction).
    :vartype preferred_tensor_id: int
    """
    def __init__(self, config=None):
        """
        Construct necessary attributes for the Inversion object.
        """
        if config is None:
            msg = "Inversion object missing keyword argument: 'config'"
            raise ValueError(msg)

        self.config = deepcopy(config)
        self.streams = None # a list of data streams
        self.moment_tensors = None
        self.preferred_tensor_id = None

    def _load_data(self):
        """
        Function to read binary SAC data

        Reads binary SAC data into a list of :class:`~obspy.core.stream.Stream` objects,
        and assigns the list to the attribute ``streams``.

        """
        files = self.config.path_to_data + "/" + self.config.station_table.station.values
        streams = [Stream() for _ in files]
        for st, file in zip(streams,files):
            for component in self.config.components:
                st.append(read("%s.%s.dat" % (file, component), format="SAC")[0])

        self.streams = streams

    def invert(self, show=False):
        """
        Launch the inversion

        Will perform multiple inversions if more than one source depth provided, and
        plot the results if attribute ``config.plot=True``

        """
        self._load_data()
        self.config.station_table.insert(loc=6+self.config.ncomp, column="weights", value=None)

        tensors = []
        for _depth in self.config.depth:
            self._single_inversion(_depth,tensors)

        # Get preferred moment tensor solution
        if len(tensors):
            vr = [mt.total_VR for mt in tensors]
            self.preferred_tensor_id = np.argmax(vr)

        #  Save all solutions
        self.moment_tensors = tensors

        if self.config.plot:
            self.plot(view="waveform", show=show)
            if len(self.config.depth) > 1:
                self.plot(view="depth", show=show)

        self._cleanup()

    def _single_inversion(self, depth, tensors):
        """
        Performs a single inversion

        Performs a single moment tensor inversion and appends the solution to
        parameter ``tensors``.

        :param depth: source depth
        :type depth: int, float
        :param tensors: moment tensor solutions
        :type tensors: list
        :return:
        """
        self._load_green(depth)
        if self.config.correlate:
            self._correlate()
        self._data_to_d()
        self._weights()

        ind = self.w != 0
        d = self.d[ind]
        G = self.G[ind, :]
        w = self.w[ind]

        Gt = (G).transpose()
        Gtd = np.dot(Gt, np.dot(np.diag(w), d))
        GtG = np.dot(Gt, np.dot(np.diag(w), G))

        # Compute the inverse GtGinv
        a = utils.gaussj(GtG, Gtd)
        m = self._a2m(a)

        # Variance reduction
        Gm = np.dot(self.G, a)
        dGm = (self.d - Gm)**2
        dd = self.d**2
        wsum = np.sum(self.w)
        var = np.sum(dGm*self.w) / wsum
        dvar = np.sum(dd*self.w) / wsum
        var /= dvar
        total_VR = (1 - var)*100

        # Station VR (unweighted)
        masked = np.ma.masked_equal(self.w,0) # only compute misfit for components included in the inversion
        dGm = masked * dGm
        dd = masked * dd

        station_VR = np.zeros((self.config.nsta,), dtype=np.float)
        for i in range(self.config.nsta):
            b = self.config.index1[i, 0]
            e = self.config.index2[i, -1]
            var = dGm[b:e]
            dvar = dd[b:e]
            var = np.sum(var)/np.sum(dvar)
            station_VR[i] = (1 - var) * 100

        # Save solution
        #kwargs = dict(depth=depth,
        #              components=self.config.components.copy(),
        #              ts=self.config.station_table.ts.values.copy(),
        #              weights=self.config.station_table.weights.values.copy(),
        #              station_VR=station_VR,
        #              total_VR=total_VR
        #              )
        kwargs = dict(
            depth=depth,
            inversion_type=self.config.inversion_type,
            components=self.config.components,
            station_VR=station_VR,
            total_VR=total_VR,
            station_table=self.config.station_table,
        )
        # Reshape for plotting
        dd, ss = self._reshape_d_Gm(Gm)
        kwargs.update(dd=dd, ss=ss)

        mt = Tensor(m*1e20, **kwargs) # Green's functions are in 1e20 dyne-cm
        mt.decompose()
        stdout = mt._get_summary()
        print(stdout) # print brief summary
        tensors.append(mt)

    def _load_green(self, depth):
        """
        Read the Green's function library

        Green's function at specified depth is assigned to attribute ``G``.

        :param depth: source depth
        :type depth: int or float
        """
        options = {"herrmann": ["_green_herrmann", ("SS", "DS", "DD", "EX")],
                   "tensor": ["_green_tensor", ("xx", "yy", "zz", "xy", "xz", "yz")],
                   }
        method = options[self.config.green][0]
        bases = options[self.config.green][1]
        files = self.config.path_to_green + "/" + self.config.station_table.station + "." + "%.4f" % depth

        # Call function based on GF format
        call_green_function = getattr(self, method)
        call_green_function(bases, files)

    def _green_tensor(self, bases, files):
        """
        Function to read Green's function in tensor format

        Construct elementary Green's functions from tensor elements
        following the formulation of Kikuchi and Kanamori, (1991).

        :param bases: the six basis Green's functions xx, yy, zz, xy, xz, yz.
        :type bases: str
        :param files: basis Green's functions (synthetics) in SAC format.
        :type files: str
        """
        #cbases = ["".join([c, basis]) for c in self.config.components for basis in bases]
        npts = self.config.station_table.npts.values
        G = np.zeros((np.sum(self.config.ncomp * npts), self.config.degree))  # G in d = Gm
        for index, file in enumerate(files):
            for i, c in enumerate(self.config.components):
                gg = np.zeros((npts[index], len(bases)))
                for j, basis in enumerate(bases):
                    gg[:, j] = read("%s.%s%s" % (file, c, basis), format="SAC")[0].data[0:npts[index]]

                b = self.config.index1[index]
                e = self.config.index2[index]
                G[b[i]:e[i], 0] = gg[:, 3]
                G[b[i]:e[i], 1] = gg[:, 0] - gg[:, 1]
                G[b[i]:e[i], 2] = gg[:, 5]
                G[b[i]:e[i], 3] = gg[:, 4]
                G[b[i]:e[i], 4] = -gg[:, 0] + gg[:, 2]
                if self.config.degree == 6:
                    G[b[i]:e[i], 5] = gg[:, 0] + gg[:, 1] + gg[:, 2]
        self.G = G

    def _green_herrmann(self, bases, files):
        """
        Function to read Green's function in Herrmann format

        A function that relates the Green's functions in the formulation of Herrmann and Wang (1985)
        to a moment tensor source.

        :param bases: the four fundamental Green's function types SS, DS, DD, EX.
        :type bases: str
        :param files: basis Green's functions (synthetics) in SAC format.
        :type files: str
        """
        # cbases order:
        # 0='ZSS', 1='ZDS', 2='ZDD', 3='ZEX'
        # 4='RSS', 5='RDS', 6='RDD', 7='REX'
        # 8='TSS', 9='TDS'

        if self.config.ncomp == 3:
            components = ["Z","R","T"]
        elif self.config.ncomp ==1 :
            components = ["Z"]
        cbases = ["".join([c, basis]) for c in components for basis in bases]
        # Remove zero value components
        try:
            cbases.remove("TDD")
            cbases.remove("TEX")
        except IndexError as e:
            print(e)

        # Read synthetics
        # Construct Green's function vector using equations 6, 7 and 8f rom Minson and Dreger, 2008 (GJI)
        # Some signs are flipped to match sign convention of basis Green's functions from RB Herrmann 2002,
        # Appendix B of Computer Programs in Seismology.
        # http://www.eas.slu.edu/eqc/eqccps.html
        npts = self.config.station_table.npts.values
        azimuth = self.config.station_table["azimuth"] * (np.pi / 180)  # azimuth in radians
        G = np.zeros((np.sum(self.config.ncomp * npts), self.config.degree))  # G in d = Gm
        for index, file in enumerate(files):
            gg = np.zeros((npts[index], len(cbases)))
            for j, basis in enumerate(cbases):
                gg[:, j] = read("%s.%s" % (file, basis), format="SAC")[0].data[0:npts[index]]

            # Transform to G
            alpha = azimuth[index]
            b = self.config.index1[index]
            e = self.config.index2[index]

            # Vertical component
            G[b[0]:e[0], 1] = gg[:, 0] * np.sin(2 * alpha)  # mxy
            G[b[0]:e[0], 2] = gg[:, 1] * np.cos(alpha)  # mxz
            G[b[0]:e[0], 4] = gg[:, 1] * np.sin(alpha)  # myz

            if self.config.degree == 5:
                G[b[0]:e[0], 0] = 0.5 * gg[:, 0] * np.cos(2 * alpha) - 0.5 * gg[:, 2]  # mxx
                G[b[0]:e[0], 3] = -0.5 * gg[:, 0] * np.cos(2 * alpha) - 0.5 * gg[:, 2]  # myy
            elif self.config.degree == 6:
                G[b[0]:e[0], 0] = (0.5 * gg[:, 0] * np.cos(2 * alpha) - 0.166667 * gg[:, 2] + 0.33333 * gg[:, 3])  # mxx
                G[b[0]:e[0], 3] = (
                        -0.5 * gg[:, 0] * np.cos(2 * alpha) - 0.166667 * gg[:, 2] + 0.33333 * gg[:, 3])  # myy
                G[b[0]:e[0], 5] = 0.33333 * gg[:, 2] + 0.33333 * gg[:, 3]  # mzz

            # Horizontal components
            if self.config.ncomp > 1:
                # mxy
                G[b[1]:e[1], 1] = gg[:, 4] * np.sin(2 * alpha)
                G[b[2]:e[2], 1] = -gg[:, 8] * np.cos(2 * alpha)
                # mxz
                G[b[1]:e[1], 2] = gg[:, 5] * np.cos(alpha)
                G[b[2]:e[2], 2] = gg[:, 9] * np.sin(alpha)
                # myz
                G[b[1]:e[1], 4] = gg[:, 5] * np.sin(alpha)
                G[b[2]:e[2], 4] = -gg[:, 9] * np.cos(alpha)  # myz

                G[b[2]:e[2], 0] = 0.5 * gg[:, 8] * np.sin(2 * alpha)  # mxx
                G[b[2]:e[2], 3] = -0.5 * gg[:, 8] * np.sin(2 * alpha)  # myy

                if self.config.degree == 5:
                    G[b[1]:e[1], 0] = 0.5 * gg[:, 4] * np.cos(2 * alpha) - 0.5 * gg[:, 6]  # mxx
                    G[b[1]:e[1], 3] = -0.5 * gg[:, 4] * np.cos(2 * alpha) - 0.5 * gg[:, 6]  # myy
                elif self.config.degree == 6:
                    G[b[1]:e[1], 0] = 0.5 * gg[:, 4] * np.cos(2 * alpha) - 0.166667 * gg[:, 6] + 0.33333 * gg[:, 7]
                    G[b[1]:e[1], 3] = -0.5 * gg[:, 4] * np.cos(2 * alpha) - 0.166667 * gg[:, 6] + 0.33333 * gg[:, 7]
                    G[b[1]:e[1], 5] = 0.33333 * gg[:, 6] + 0.33333 * gg[:, 7]  # mzz

        self.G = G

        if ''.join(self.config.components) == "ZNE":
            self._rotate_rt_to_ne()

    def _rotate_rt_to_ne(self):
        """
        Rotate horizontal component Green's functions
        """
        for i in range(self.config.nsta):
            rad_start = self.config.index1[i, 1]
            rad_end = self.config.index2[i, 1]
            tan_start = self.config.index1[i, 2]
            tan_end = self.config.index2[i, 2]
            r = self.G[rad_start:rad_end]
            t = self.G[tan_start:tan_end]
            n, e = utils.rotate_rt2ne(r, t, self.config.station_table.azimuth[i])
            self.G[rad_start:rad_end] = n
            self.G[tan_start:tan_end] = e

    @staticmethod
    def _save_fundamental_fault_types(syn, gg, b, e, degree, ncomp):
        for i in range(ncomp):
            syn[b[i]:e[i], 0] = gg[:, 4 * i]  # 0:ZSS, 4:RSS, 8:TSS
            syn[b[i]:e[i], 1] = gg[:, 4 * i + 1]  # 1:ZDS, 5:RDS, 9:TDS
            if i < 2:
                syn[b[i]:e[i], 2] = gg[:, 4 * i + 2]  # 2:ZDD, 6:RDD
                if degree == 6:
                    syn[b[i]:e[i], 3] = gg[:, 4 * i + 3]  # 3:ZEX, 7:REX

    def _correlate(self, abs_max=True):
        # cross-correlate data and Green's functions for best time shift (in samples)
        lenc = self.streams[0][0].stats.npts - self.config.station_table.npts.values + 1
        ts = self.config.station_table.ts.values
        for index in range(self.config.nsta):
            b = self.config.index1[index]
            e = self.config.index2[index]
            corr = 0.
            for j in range(self.config.degree):
                c3 = np.zeros(lenc[index])
                for i, component in enumerate(self.config.components):
                    c = utils.xcorr(self.streams[index].select(component=component)[0].data, self.G[b[i]:e[i], j])
                    c3 += c
                c3 /= self.config.ncomp
                position = np.argmax(np.abs(c3) if abs_max else c3)
                value = np.abs(c3[position])
                # Update time shift
                if value > corr:
                    corr = value
                    shift = position
            ts[index] = shift

        print("Data and Green's functions xcorr:")
        print(self.config.station_table[["station", "ts"]])

    def _data_to_d(self):
        # construct data vector according to time shift and sample size
        d = np.zeros(np.sum(self.config.ncomp * self.config.station_table.npts.values), dtype=np.float)
        obs_b = self.config.station_table.ts.values
        obs_e = obs_b + self.config.station_table.npts.values
        for i in range(self.config.nsta):
            b = self.config.index1[i]
            e = self.config.index2[i]
            for d_b, d_e, component in zip(b, e, self.config.components):
                d[d_b:d_e] = self.streams[i].select(component=component)[0].data[obs_b[i]:obs_e[i]]

        self.d = d

    def _weights(self):
        # Compute weights
        if self.config.weight == "none":
            weights = 1
        elif self.config.weight == "distance":
            weights = self.config.station_table.distance / np.min(self.config.station_table.distance)
        elif self.config.weight == "variance":
            wei = np.zeros((self.config.nsta,))
            for i in range(self.config.nsta):
                b = self.config.index1[i, 0]
                e = self.config.index2[i, -1]
                wei[i] = 1 / np.sum(self.d[b:e] ** 2)
            weights = wei / np.sum(wei)

        self.config.station_table.weights = weights
        w = self.config.station_table[list(self.config.components)].mul(self.config.station_table.weights, axis=0).to_numpy()
        w = np.repeat(w, np.repeat(self.config.station_table.npts.to_numpy(), self.config.ncomp))
        self.w = w

    def _a2m(self, a):
        """
         aij coefficients to moment tensor elements
        """
        # Output order: Mxx, Myy, Mzz, Mxy, Mxz, Myz
        m = np.zeros((6,),dtype=np.float)

        if self.config.green == "tensor":
            m[3] = a[0]
            m[4] = a[3]
            m[5] = a[2]
            if self.config.degree == 6:
                m[0] = a[1] - a[4] + a[5]
                m[1] = -a[1] + a[5]
                m[2] = a[4] + a[5]
            elif self.config.degree == 5:
                m[0] = a[1] - a[4]
                m[1] = -a[1]
                m[2] = a[4]
        elif self.config.green == "herrmann":
            m[0] = a[0]
            m[1] = a[3]
            m[3] = a[1]
            m[4] = a[2]
            m[5] = a[4]
            if self.config.degree == 6:
                m[2] = a[5]
            elif self.config.degree == 5:
                m[2] = -(a[0] + a[3])

        return m

    def _reshape_d_Gm(self, Gm):
        npts = self.config.station_table.npts.values
        dd = [None for _ in npts]
        ss = [None for _ in npts]
        for i in range(self.config.nsta):
            dd[i] = np.reshape(self.d[self.config.index1[i, 0]:self.config.index2[i, -1]],
                               (self.config.ncomp, npts[i])).T
            ss[i] = np.reshape(Gm[self.config.index1[i, 0]:self.config.index2[i, -1]],
                               (self.config.ncomp, npts[i])).T

        return dd, ss

    def get_preferred_tensor(self):
        """
        Returns the preferred moment tensor solution

        A function that returns the solution with the highest variance reduction (goodness-of-fit
        between data and synthetic waveforms).

        :return: the preferred moment tensor solution.
        :rtype: a :class:`~mttime.core.tensor.Tensor` object.
        """
        if self.preferred_tensor_id is None:
            return None
        return self.moment_tensors[self.preferred_tensor_id]

    def write(self, option=None):
        """
        Write inversion results to file

        Save all solutions or only the preferred solution to file.
        The output file name format is ``d[depth].mtinv.out``.

        :param option: option to write all solutions or only the preferred solution. Default is
            ``None``. If set to ``preferred`` only the solution with the highest VR will be saved.
        :type option: str
        """
        # Write Configure object to file
        # self.config.write()

        # Write detailed moment tensor solution to a text file
        if option == "preferred":
            self.get_preferred_tensor().write()
        else:
            for tensor in self.moment_tensors:
                tensor.write()

    def plot(self, **kwargs):
        """
        Plot inversion results

        Various options available to display the results.

        :param view: type of figure to produce. Default ``waveform`` creates the
            standard figure with focal mechanisms and waveform fits.
            ``depth`` shows the focal mechanism and moment magnitude as a
            function of depth. ``map`` plots stations and focal mechanisms in map view.
            ``lune`` plots the moment tensor source-type on a lune.
        :type view: str
        :param show: If ``True`` will display figure after plotting and not save image to file,
            default is ``False``.
        :type show: bool
        :param format: figure file format, default is ``"eps"``.
        :type format: str
        :param option: Optional parameter if view is set to ``waveform``. The default plots all
            solutions. Set to ``preferred`` to plot only the preferred solution.
        :type option: str, optional

        """
        view = kwargs.get("view", "waveform")
        show = kwargs.get("show", False)
        format = kwargs.get("format","eps")
        if view == "waveform":
            option = kwargs.get("option", None)
            from mttime.imaging.source import plot_waveform_fits
            if option == "preferred":
                tensors = [self.get_preferred_tensor()]
            else:
                tensors = self.moment_tensors
            for tensor in tensors:
                plot_waveform_fits(tensor, show, format)
        elif view == "map":
            if self.config.event is None:
                print("Event origin is missing, cannot plot in map view.")
            else:
                from mttime.imaging.source import beach_map
                m = self.get_preferred_tensor().m

                args = (
                    self.config.event,
                    self.config.station_table.longitude.values,
                    self.config.station_table.latitude.values,
                    self.config.station_table.distance.values,
                    self.config.station_table[self.config.components].sum(axis=1).astype(bool).values,
                    show,
                    format,
                )
                beach_map(m, *args)
        elif view == "depth":
            from mttime.imaging.source import beach_mw_depth
            beach_mw_depth(self.moment_tensors, self.config.event, show, format)
        elif view == "lune":
            if self.config.degree != 6:
                msg = "Full moment tensor is required to generate source-type plot."
                raise ValueError(msg)
            from mttime.imaging.source import plot_lune
            mt = self.get_preferred_tensor()
            m = mt.m
            gamma, delta = mt.lune
            plot_lune(m, gamma, delta, show, format)
        else:
            raise KeyError("'view=%s' is not supported."%view)

    def _cleanup(self):
        del self.d
        del self.w

    def __str__(self):
        ret = "\n".join(
            [self.config.__str__(),
             "\n| PREFERRED SOLUTION |",
             self.get_preferred_tensor().__str__()]
        )

        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
