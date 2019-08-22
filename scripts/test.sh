maindir=/home/kumarv/khand035/Projects/MINT/river-depth-estimation/scripts
ctr=0
for cdir in $maindir/Boxes/*.txt
do
  	boxname=$(basename $cdir)
	echo $maindir/Boxes/$boxname
    	ctr=$((ctr+1))
done
