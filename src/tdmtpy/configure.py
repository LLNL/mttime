# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea@llnl.gov)
"""
Set global tdmtpy configuration

.. rubric:: Example

.. code-block:: python

   >>> import tdmtpy
   >>> # read input file mtinv.in
   >>> config = tdmtpy.Configure(path_to_file="mtinv.in")

"""

import os
import warnings

import numpy as np
import pandas as pd


class Configure(object):
    """
    Configure object for :class:`~tdmtpy.inversion.Inversion`

    Sets up the moment tensor inverse routine. ``**kwargs`` can be provided
    either in ``path_to_file`` or during class instantiation.
    Any ``**kwargs`` provided during class instantiation will override the values
    from file, and if any values are left blank in the file the default value will be used instead.

    :param path_to_file: path to input file containing headers and station information.
        Directory will become the project root directory. Default is ``"./mtinv.in"``.
    :type path_to_file: str
    :param datetime: event origin time.
    :type datetime: str, optional
    :param longitude: event longitude.
    :type longitude: float, optional
    :param latitude: event latitude.
    :type latitude: float, optional
    :param depth: source depths to invert.
    :type depth: float, int, list of floats/ints
    :param path_to_data: path to data files, relative to root directory.
        Defaults is ``"./"``.
    :type path_to_data: str
    :param path_to_green: path to Green's function files, relative to root directory.
        Defaults is ``"./"``.
    :type path_to_green: str
    :param green: Green's function format, options are ``"herrmann"`` or ``"tensor"``.
        Defaults to ``"herrmann"``.
    :type green: str
    :param components: waveform components, options are ``"Z"`` for vertical component,
        or ``"ZRT"`` for three-component data in vertical, radial and transverse components, and
        ``"ZNE"`` for vertica, north and east.
        Defaults to ``"ZRT"``.
    :type components: list of str
    :param degree: degrees of freedom allowed in the inversion, options are ``5`` for deviatoric
        or ``6`` for full. Defaults to ``5``.
    :type degree: int
    :param weight: data weights, options are ``"none"``, ``"distance"`` or ``"variance"``
        for no weights, inverse distance, or inverse variance, respectively.
        Defaults to ``"none"``.
    :type weight: str
    :param plot: If ``True`` will plot the solution and waveform fits. Default is ``False``.
    :type plot: int, bool
    :param correlate: Flag to cross-correlate data and Green's functions
        for best time shift (in time points). Default is ``False``.
    :type correlate: int, bool
    """
    def __init__(self, path_to_file="mtinv.in", **kwargs):
        # dict types
        types = dict(datetime=str,
                     longitude=float,
                     latitude=float,
                     depth=str,  # comma-delimited
                     path_to_data=str,
                     path_to_green=str,
                     green=str,
                     components=str,
                     degree=int,
                     weight=str,
                     plot=int,
                     correlate=int
                    )
        # Station table dtypes
        df_dtypes = {"station": str,
                     "distance": np.float,
                     "azimuth": np.float,
                     "ts": np.int,
                     "npts": np.int,
                     "dt": np.float,
                     "used": str,
                     "longitude": np.float,
                     "latitude": np.float,
                     # Optional
                     "filter": str,
                     "nc": np.int,
                     "np": np.int,
                     "lcrn": np.float,
                     "hcrn": np.float,
                     "model": str
                    }

        self.__dict__["types"] = types
        self.__dict__["df_dtypes"] = df_dtypes

        # Read keywords from file first
        header, home = self._read(path_to_file, types)

        # Override parameters from file if provided kwargs during class instantiation
        # then set class attributes
        self._set_attributes(home, header, **kwargs)

        # Read station table
        self._update_table_index(path_to_file,df_dtypes)

    @property
    def types(self):
        """
        dict of types for ``**kwargs``, read only
        """
        return self.__dict__["types"]

    @property
    def df_dtypes(self):
        """
        dict of dtypes for station table, read only
        """
        return self.__dict__["df_dtypes"]

    @staticmethod
    def _read(path_to_file, types):
        """
        Read inversion parameters from a text file.

        Parse input text file to a python dictionary and returns the python dictionary.

        :param path_to_file: file to read
        :type path_to_file: str
        :param types: keyword arguments
        :type types: dict
        :return: a dictionary of parameters for a single :class:`~tdmtpy.configure.Configure` object
        :rtype: dict
        """

        # Check file
        path_to_file = os.path.abspath(path_to_file)
        if not isinstance(path_to_file, str):
            raise TypeError("File name must be a string.\n")
        try:
            with open(path_to_file):
                pass
        except IOError:
            print("Cannot open '%s'" % path_to_file)

        header = dict()

        # Read keyword arguments from file
        with open(path_to_file, "r") as f:
            for key, parse in types.items():
                try:
                    header[key] = parse(next(f).split()[1])
                except IndexError:
                    msg = "'%s' missing in %s." % (key, path_to_file)
                    warnings.warn(msg)
                    continue
        if "depth" in header:
            header["depth"] = [float(depth) for depth in header["depth"].split(",") if depth]

        # project root directory
        home = path_to_file.rsplit("/", 1)
        if len(home) > 1:
            home = home[0]
        else:
            home = "."

        return header, home

    def write(self, fileout="config.out"):
        """
        Function to write inversion configuration to a file

        Write inverse routine parameters and station table to a file.

        :param fileout: output file name. Default is ``"config.out"``.
        :type fileout: str

        .. rubric:: Example
        .. code-block:: python

           >>> config = Configure()
           >>> config.write(fileout="configure.out")
        """
        with open(fileout, "w") as f:
            f.write(self.__str__())

    def _set_attributes(self, home, header, **kwargs):
        """
        Set class attributes

        :param home: root directory
        :type home: str
        :param header: a dictionary of parameters
        :type header: dict
        :param kwargs: inputs from console
        ;type kwargs: dict
        """
        # Over-ride parameters and set type
        for key, parse in self.types.items():
            if key in kwargs:
                if key == "depth":
                    pass
                else:
                    kwargs[key] = parse(kwargs[key])
            else:
                try:
                    kwargs[key] = header[key]
                except KeyError:
                    continue

        # Optional event attribute
        event = dict()
        event["datetime"] = kwargs.get("datetime","")
        event["longitude"] = kwargs.get("longitude",np.nan)
        event["latitude"] = kwargs.get("latitude",np.nan)
        self.event = event

        # Required attributes
        self.depth = []
        self._set_depth(kwargs["depth"])
        self.path_to_data = os.path.abspath(
            "/".join([home, kwargs.get("path_to_data", ".")]))
        self.path_to_green = os.path.abspath(
            "/".join([home, kwargs.get("path_to_green", ".")]))
        kwargs["green"] = kwargs.get("green", "herrmann")
        if kwargs["green"].lower() not in ("herrmann", "tensor"):
            raise ValueError("green not supported.")
        self.green = kwargs["green"].lower()
        kwargs["components"] = kwargs.get("components", "ZRT")
        if any(c == kwargs["components"].upper() for c in ("Z", "ZRT", "ZNE")):
            if kwargs["components"].upper() == "ZNE":
                self.rotate_to_zne = True
            else:
                self.rotate_to_zne = False
            self.components = list(kwargs["components"].upper())
        else:
            raise ValueError("components not supported.")
        self.degree = kwargs.get("degree", 5)
        kwargs["weight"] = kwargs.get("weight", "none")
        if kwargs["weight"].lower() not in ("none", "distance", "variance"):
            raise ValueError("weight not supported.")
        self.weight = kwargs["weight"].lower()
        self.plot = bool(kwargs.get("plot", False))
        self.correlate = bool(kwargs.get("correlate", False))

        # Additional attributes
        self.inv_type = {5: "Deviatoric", 6: "Full"}[self.degree]

    def _set_depth(self, depth):
        """
        Set depths

        :param depth: source depths to invert
        :type depth: float, int, list of float/ints
        :return:
        """
        if isinstance(depth, (int, float)):
            self.depth = [depth]
        elif isinstance(depth, list):
            if len(depth) < 1:
                msg = "Depth is empty."
                raise IndexError(msg)
            for _i in depth:
                # Make sure each item in the list is a number.
                if not isinstance(_i, (int, float)):
                    msg = "Depth must be a list of numbers."
                    raise TypeError(msg)
            self.depth = depth
        else:
            msg = "Depth must be a number or a list of numbers."
            raise TypeError(msg)

    def _update_table_index(self,path_to_file,df_dtypes):
        """
        Constructs the station table from input file

        :param path_to_file: path to input file containing headers and station information.
        :type path_to_file: str
        :param df_dtypes: dtypes for station table
        :type df_dtypes: dict
        """

        # Read station table
        df = pd.read_table(path_to_file, sep=r"\s+", dtype=df_dtypes, skiprows=12)
        self.df = df

        self.nsta = len(self.df.index)
        self.ncomp = len(self.components)

        # Components to invert
        df = self.df["used"].apply(lambda x: pd.Series(list(x)))

        if len(df.columns) < self.ncomp:
            df = pd.concat([df] * (self.ncomp), axis=1, ignore_index=True)
        else:
            df = df.iloc[:, [i for i in range(self.ncomp)]]

        df.columns = self.components
        location = 6
        for component in self.components:
            self.df.insert(loc=location, column=component, value=df[component])
            self.df[component].fillna(self.df.Z, inplace=True)
            location += 1

        self.df.drop(columns="used", axis=1, inplace=True)
        self.df[self.components] = self.df[self.components].astype(np.int)

        # Indexing linear equations
        index2 = np.cumsum(np.repeat(self.df.npts.to_numpy(), self.ncomp), dtype=np.int)
        index1 = np.zeros(index2.shape, dtype=np.int)
        index1[1::] = index2[0:-1]
        self.index1 = index1.reshape(self.nsta, self.ncomp)
        self.index2 = index2.reshape(self.nsta, self.ncomp)

    def __str__(self):
        keys = ("event",
                "depth",
                "green",
                "components",
                "degree",
                "weight",
                "plot",
                "correlate"
                )
        f = "{0:>12}: {1}\n"
        ret = "".join([f.format(key, str(getattr(self, key))) for key in keys])
        ret = "\n".join([ret, self.df.to_string(index=False), "\n"])

        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
