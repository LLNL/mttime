
tdmtpy
======

tdmtpy (Time Domain Moment Tensor Inversion in Python) is a python package developed for time domain inversion of complete seismic waveform data
to obtain the seismic moment tensor. It supports deviatoric and full moment tensor inversions,
and 1-D and 3-D basis Green's functions.

Requirements
------------
The package was developed on python 3.7 and 3.8, and is running and tested on Mac OSX.

* ObsPy 1.0.* and its dependencies
* pandas 1.0.*

Installation
------------

* Create a Python environment
* Install ObsPy and pandas
* Make sure you have `cloned the repository <https://github.com/LLNL/tdmtpy>`_
* Install tdmtpy

I recommend installing Python via `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
or `Anaconda <https://docs.anaconda.com/anaconda/install/>`_. Choose Miniconda for a lower footprint.
Then follow the instructions on their sites to install
`ObsPy <https://github.com/obspy/obspy/wiki/Installation-via-Anaconda>`_
and `pandas <https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html>`_
for your given platform.

Download tdmtpy and install it from source. If you installed Python via conda make sure you activate
the environment where ObsPy and pandas are installed.

.. code-block:: bash

   # Activate environment
   conda activate your_environment

   # Build and install tdmtpy
   git clone https://github.com/LLNL/tdmtpy
   cd tdmtpy
   python setup.py install


Finally, if you want to run the tutorials you will need to install `Jupyter Notebook <https://jupyter.org/install>`_.

Usage
-----

You can execute the package from the command line or run it interactively in the Python console.
In either case you will need to provide an input file that contains information about the
file structure and parameters to set up the inverse routine.
If no file is given the code will look for a file named `mtinv.in` in the current working directory.

Executing the package from command line will launch the inversion,
save and plot the result to file:

.. code-block:: bash

   tdmtpy-run mtinv.in

The equivalent in the Python console:

.. code-block:: python

   import tdmtpy
   config = tdmtpy.Configure(path_to_file="mtinv.in")
   mt = tdmtpy.Inversion(config=config)
   mt.invert()
   mt.write()

Refer to the :ref:`user_guide` for more details.

Tutorials
---------
Check out the Jupyter Notebooks for a working example.

License
-------
`tdmtpy` is distributed under the terms of LGPL-3.0 license. All new contributions must be made under the LGPL-3.0 license.

SPDX-License-Identifier: LGPL-3.0

LLNL-CODE-814839
