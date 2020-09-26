#!/bin/zsh
## SCRIPT TO GENERATE EXAMPLE DATA AND GREEN'S FUNCTIONS ##

depth=10.0000
dt=1
npts=256
M0=1e+20 # tensor GF
M0_mech=1e+22
lcrn=0.05
hcrn=0.1

# Velocity model
cat > gil7.d << EOF
MODEL.01
GIL7
ISOTROPIC
KGS
FLAT EARTH
1-D
CONSTANT VELOCITY
LINE08
LINE09
LINE10
LINE11
 H(KM)   VP(KM/S)   VS(KM/S) RHO(GM/CC)         QP         QS   ETAP       ETAS      FREFP      FREFS
1.0000       3.20       1.50       2.28     600.00     300.00   0.00       0.00       1.00       1.00
2.0000       4.50       1.40       2.28     600.00     300.00   0.00       0.00       1.00       1.00
1.0000       4.80       2.78       2.58     600.00     300.00   0.00       0.00       1.00       1.00
1.0000       5.51       3.18       2.58     600.00     300.00   0.00       0.00       1.00       1.00
12.000       6.21       3.40       2.68     600.00     300.00   0.00       0.00       1.00       1.00
8.0000       6.89       3.98       3.00     600.00     300.00   0.00       0.00       1.00       1.00
0.0000       7.83       4.52       3.26     600.00     300.00   0.00       0.00       1.00       1.00
EOF

dirout=greens # green's functions
stat=(BK.FARB.00 BK.SAO.00 BK.CMB.00 BK.MNRC.00)
dist=(110 120 123 132)
azi=(263 167 78 333)
nsta=`echo $dist | awk 'END{print (NF+1)}'`

mkdir -p $dirout

\rm dfile
for ((i=1; i<$nsta; i++)); do
	echo "$dist[$i] $dt $npts 0.0 0.0" >> dfile
done


\rm hspec96.* file96
for d in `echo $depth`; do
	hprep96 -M gil7.d -d dfile -HS $d -HR 0 -EQEX
	hspec96
	hpulse96 -D -i > file96
	
	## Green's functions ##
	nsta=`awk 'END{print NR+1}' dfile`
		for ((i=1; i<$nsta; i++)); do
			
			num=`printf "%03d\n" $i`
			ofile=$dirout/$stat[$i].$depth
		# Langston/Herrmann type GF
		f96tosac -B file96

		knetwk=`echo $stat[$i] | awk -F'.' '{print $1}'`
		kstnm=`echo $stat[$i] | awk -F'.' '{print $2}'`
		khole=`echo $stat[$i] | awk -F'.' '{print $3}'`
sac << EOF
read B${num}08TSS.sac B${num}05TDS.sac B${num}07RSS.sac B${num}04RDS.sac B${num}02RDD.sac
ch knetwk $knetwk
ch kstnm $kstnm
ch khole $khole
bp co $lcrn $hcrn p 2
rmean
write $ofile.TSS $ofile.TDS $ofile.RSS $ofile.RDS $ofile.RDD 

read B${num}06ZSS.sac B${num}03ZDS.sac B${num}01ZDD.sac B${num}10REX.sac B${num}09ZEX.sac
ch knetwk $knetwk
ch kstnm $kstnm
ch khole $khole
bp co $lcrn $hcrn p 2
rmean
write $ofile.ZSS $ofile.ZDS $ofile.ZDD $ofile.REX $ofile.ZEX
quit
EOF
		
		# Create single couple Green's functions (required for 3D)
		for IJ in XX XY XZ YY YZ ZZ; do           
			#b=`echo "$IJ" | awk '{print tolower($1)}'`
# ZRT
fmech96 -$IJ $M0 -A $azi[$i] -ROT < file96 | f96tosac -B
sac << EOF
read B${num}03T00.sac B${num}02R00.sac B${num}01Z00.sac
bp co $lcrn $hcrn p 2
rmean
ch knetwk $knetwk
ch kstnm $kstnm
ch khole $khole
ch kevnm $b
ch file 1 kcmpnm T
ch file 2 kcmpnm R
ch file 3 kcmpnm Z
write $ofile.T$IJ $ofile.R$IJ $ofile.Z$IJ
quit
EOF
# ZNE
#fmech96 -$IJ $M0 -A $azi[$i] < file96 | f96tosac -B
#sac << EOF
#read B${num}03E00.sac B${num}02N00.sac B${num}01Z00.sac
#bp co $lcrn $hcrn p 2
#rmean
#ch knetwk $knetwk
#ch kstnm $kstnm
#ch khole $khole
#ch kevnm $b
#ch file 1 kcmpnm E
#ch file 2 kcmpnm N
#ch file 3 kcmpnm Z
#write $ofile.e$b $ofile.n$b $ofile.z$b
#quit
#EOF
		done # 3D
		
############### Data ############### 
		dir1=earthquake
		mkdir -p $dir1
fmech96 -D 67 -S 123 -R 45 -M0 $M0_mech -A $azi[$i] -ROT < file96 | f96tosac -B
sac << EOF
cuterr fillz
cut o -30 o 250
read B${num}03T00.sac B${num}02R00.sac B${num}01Z00.sac
bp co $lcrn $hcrn p 2
rmean
ch knetwk $knetwk
ch kstnm $kstnm
ch khole $khole
ch file 1 kcmpnm T
ch file 2 kcmpnm R
ch file 3 kcmpnm Z
write $dir1/$stat[$i].T.dat $dir1/$stat[$i].R.dat $dir1/$stat[$i].Z.dat
quit
EOF

#fmech96 -D 67 -S 123 -R 45 -MW 4 -A $azi[$i] < file96 | f96tosac -B
#sac << EOF
#cuterr fillz
#cut o -30 o 250
#read B${num}03E00.sac B${num}02N00.sac B${num}01Z00.sac
#bp co $lcrn $hcrn p 2
#rmean
#ch knetwk $knetwk
#ch kstnm $kstnm
#ch khole $khole
#ch file 1 kcmpnm E
#ch file 2 kcmpnm N
#ch file 3 kcmpnm Z
#write $dir1/$stat[$i].e.dat $dir1/$stat[$i].n.dat $dir1/$stat[$i].z.dat
#quit
#EOF

		dir2=explosion
		mkdir -p $dir2
		fmech96 -E -M0 $M0_mech -A $azi[$i] -ROT < file96 | f96tosac -B
sac << EOF
cuterr fillz
cut o -30 o 250
read B${num}03T00.sac B${num}02R00.sac B${num}01Z00.sac
bp co $lcrn $hcrn p 2
rmean
ch knetwk $knetwk
ch kstnm $kstnm
ch khole $khole
ch file 1 kcmpnm T
ch file 2 kcmpnm R
ch file 3 kcmpnm Z
write $dir2/$stat[$i].T.dat $dir2/$stat[$i].R.dat $dir2/$stat[$i].Z.dat
quit
EOF
#fmech96 -E -MW 3.5 -A $azi[$i] < file96 | f96tosac -B
#sac << EOF
#cuterr fillz
#cut o -30 o 250
#read B${num}03E00.sac B${num}02N00.sac B${num}01Z00.sac
#ch kstnm $stat[$i]
#bp co $lcrn $hcrn p 2
#rmean
#ch knetwk $knetwk
#ch kstnm $kstnm
#ch khole $khole
#ch file 1 kcmpnm E
#ch file 2 kcmpnm N
#ch file 3 kcmpnm Z
#write $dir2/$stat[$i].e.dat $dir2/$stat[$i].n.dat $dir2/$stat[$i].z.dat
#quit
#EOF
		dir3=composite
		mkdir -p $dir3
		fmech96 -XX 6e+21 -YY 6e+21 -ZZ 8e+21 -XY 1e+21 -XZ 1e+21 -YZ 1e+21 -A $azi[$i] -ROT < file96 | f96tosac -B
sac << EOF
cuterr fillz
cut o -30 o 250
read B${num}03T00.sac B${num}02R00.sac B${num}01Z00.sac
ch kstnm $stat[$i]
bp co $lcrn $hcrn p 2
rmean
ch knetwk $knetwk
ch kstnm $kstnm
ch khole $khole
ch file 1 kcmpnm T
ch file 2 kcmpnm R
ch file 3 kcmpnm Z
write $dir3/$stat[$i].T.dat $dir3/$stat[$i].R.dat $dir3/$stat[$i].Z.dat
quit
EOF
#fmech96 -XX 6e+21 -YY 6e+21 -ZZ 8e+21 -XY 1e+21 -XZ 1e+21 -YZ 1e+21 -A $azi[$i] < file96 | f96tosac -B
#sac << EOF
#cuterr fillz
#cut o -30 o 250
#read B${num}03E00.sac B${num}02N00.sac B${num}01Z00.sac
#ch kstnm $stat[$i]
#bp co $lcrn $hcrn p 2
#rmean
#ch knetwk $knetwk
#ch kstnm $kstnm
#ch khole $khole
#ch file 1 kcmpnm E
#ch file 2 kcmpnm N
#ch file 3 kcmpnm Z
#write $dir3/$stat[$i].e.dat $dir3/$stat[$i].n.dat $dir3/$stat[$i].z.dat
#quit
#EOF
	done # stations
done # depth

\rm *.sac
