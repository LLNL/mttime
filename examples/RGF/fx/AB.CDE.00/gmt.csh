#!/bin/csh

gmtset PLOT_DEGREE_FORMAT D
gmtset COLOR_MODEL HSV
gmtset PAPER_MEDIA letter
gmtset PAGE_ORIENTATION portrait
gmtset MEASURE_UNIT inch

# Region will need to be adjusted based on etree/grid values
set REGION = -118.009/-117.881/34.991/35.0988

set SCALE = 6.0

# These commands are good if you have access to 
# a topography database file for the region modeled 
# Note:  if you uncomment these, adjust the -O -K, etc.
 #######################################################
#grdraster 2 -R$REGION -I0.5m -Gwpp_topo.grd
#grdgradient wpp_topo.grd -Gwpp_topo_shade.grd -A270 -Nt -M 
#grd2cpt wpp_topo.grd -Ctopo -Z >! wpptopo.cpt
#grdimage wpp_topo.grd -R$REGION -JM$SCALE -Cwpptopo.cpt -Iwpp_topo_shade.grd -P -K >! plot.ps
 #######################################################
pscoast -R$REGION -JM$SCALE -Bf0.025a0.05 -Dfull -S100,200,255 -A2000 -W3 -N1t3 -N2t2a -K >! plot.ps

# computational grid region...
psxy -R$REGION -JM$SCALE -W10/255/255/0ta -O -K <<EOF>> plot.ps
-118 35
-118 35.0898
-117.89 35.0898
-117.89 35
-118 35
EOF

#SG boundary: 
psxy -R$REGION -JM$SCALE -W10/255/255/0ta -O -K <<EOF>> plot.ps
-117.967 35.0269
-117.923 35.0269
-117.923 35.0629
-117.967 35.0629
-117.967 35.0269
EOF

# Sources... 
cat << EOF >! event.d
-117.956 35.0359 EVENT-NAME  CB
EOF
psxy -R -J -O -K -Sc0.1 -Gred -W0.25p event.d >> plot.ps
awk '{print $1, $2, 12, 1, 9, $4, $3}' event.d | pstext -R -J -O -D0.2/0.2v -Gred -N -K >> plot.ps

# Stations... 
cat << EOF >! stations.d 
-117.934 35.0539 source1 CB
EOF

# plot station names
psxy -R -J -O -K -St0.1 -Gblue -W0.25p stations.d >> plot.ps
awk '{print $1, $2, 12, 1, 9, $4, $3}' stations.d | pstext -R -J -O -Dj0.3/0.3v -Gblue -N >> plot.ps

/bin/mv plot.ps fx.in.ps
