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

# +
import os
import sys
sys.path.insert(0, os.path.abspath('/Users/chiang4/PycharmProjects/mttime_github/src'))

import mttime
import numpy as np
import pandas as pd
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
# -

config = mttime.Configure(path_to_file="mtinv.in")
tdmt = mttime.Inversion(config=config)
tdmt.invert()

view = "waveform"
tdmt.plot(view=view,show=True)
tdmt.plot(view=view,show=False)

view = "map"
tdmt.plot(view=view,show=True)
tdmt.plot(view=view,show=False)

view = "lune"
tdmt.plot(view=view,show=True)
tdmt.plot(view=view,show=False)

view = "depth"
tdmt.plot(view=view,show=True)
tdmt.plot(view=view,show=False)

tdmt.write()
