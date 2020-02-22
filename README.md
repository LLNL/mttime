# tdmtpy
tdmtpy is a software tool developed for time domain inversion of complete seismic waveform data to obtain the seismic moment tensor. It supports both deviatoric and full moment tensor inversions as well as 1-D and 3-D basis Green's functions.

### Usage
1. From command line: tdmt [input_file], if no input file is specified code will look for default input file "./mtinv.in"

2. Below is an example input file **mtinv.in**, check :Class: `tdmtpy.Header` and :Class: `tdmtpy.Station` for more details on the input parameters.
```
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
```

3. Data and Green's functions are in SAC binary format, and are corrected for instrument response, filtered, and decimated. File name coventions are described below:
   - Data: [name].[components].dat
	   - Name: station name in input file
	   - Components: t, r, or z
		- Examples: BK.CMB.00.z.dat, BK.CMB.00.t.dat
    
   - Green's Functions: [NAME].[DEPTH].[GF_NAME]
     - NAME: station name in inputfile
	  - DEPTH: source depth with four significant digits
	  - COMPONENTS: t, r or z
     - GF_NAME: depends on the Green's function format.
       1. herrmann format has 10 GFs: tss tds rss rds rdd zss zds zdd rex zex
			 2. tensor format has 18 GFs (if using all three components): zxx, zyy, zzz, zxy, zxz, zyz, etc.
     - Examples:
       1. Herrmann format: BK.CMB.00.10.0000.zds
			 2. Tensor format: BK.CMB.00.10.0000.zxy

4. Two output files are created "mtinv.out" and "max.mtinv.out" after running the code.
	mtinv.out = moment tensor depth search results, best solution on the second line (after header)
	max.mtinv.out = best solution with the highest VR, includes additional station information

5. If plot = 1 code will generate figures (e.g. figure0.pdf, figure1.pdf, etc.) with beach balls and waveform fits plotted

License
tdmtpy is distributed under the terms of BSD-3 license.
All new contributions must be made under the BSD-3 license.

SPDX-License-Identifier: (BSD-3)

LLNL-CODE-805542
