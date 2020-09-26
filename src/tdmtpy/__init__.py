# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
tdmtpy: Time Domain Moment Tensor Inversion in Python
=====================================================

tdmtpy is an open-source python package developed for time domain inversion of complete seismic waveform data
to obtain the seismic moment tensor.

:copyright:
    Copyright (c) 2020, Lawrence Livermore National Security, LLC
:license:
    BSD 3-Clause License
"""

# Generic release markers:
# X.Y
# X.Y.Z # bug fix and minor updates

# dev branch marker is "X.Y.devN" where N is an integer.
# X.Y.dev0 is the canonical version of X.Y.dev

__version__ = "1.0.0"

from .configure import Configure
from .inversion import Inversion
