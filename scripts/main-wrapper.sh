export PATH=/home/kumarv/khand035/.conda/envs/tf_cpu_clone_jup/bin:/home/kumarv/khand035/.conda/envs/gdalenv/bin/:/home/kumarv/khand035/Projects/CodeLibrary/pycharm-community-2018.2.5/bin:$PATH
export LD_LIBRARY_PATH=/home/kumarv/khand035/.conda/envs/tf_cpu_clone_jup/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/kumarv/khand035/.conda/envs/gdalenv/lib/python2.7/site-packages/
export KERAS_BACKEND=tensorflow

#python GEEScript.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/GageBoxes2_wbd.shp $2 $3 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/basins_wgs84_dissovle.shp $1 $5

#python GEEClipDEM.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/GageBoxes2_wbd.shp /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/dem_wgs84_sqcut.tif credentials.txt /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/basins_wgs84_dissovle.shp $1 $5 dem.tiff

#python GEEClipDEM.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/GageBoxes2_wbd.shp /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/JRC-Recurrence-Pongo.tif credentials.txt /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/basins_wgs84_dissovle.shp $1 $5 jrc.tiff

for clipdir in /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$4*/ ; 
do
	rm -rf $clipdir/M*.tif
	python ProcessZip_wrapper.py $clipdir $(basename $clipdir).txt 1

	#rm -rf $clipdir/F*.tif
	#python MergeMaps.py $clipdir

	#python ManualMapSelection.py $clipdir 

	#rm $clipdir/*.png
	#rm $clipdir/*.csv
	#python EstimateRiverDepth.py $clipdir

	#cp $clipdir/*.png /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/Figures/
	#cp $clipdir/*.csv /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/Figures/
	#cp $clipdir/*-viz.csv ./../CSVs
	#cd ..
	#git pull https://ankushkhandelwal:Hermionilv_s23@github.com/ankushkhandelwal/river-depth-estimation.git
	#git add .
	#git commit -m "auto push"
	#git push https://ankushkhandelwal:Hermionilv_s23@github.com/ankushkhandelwal/river-depth-estimation.git
	#cd scripts
	python catalog.py  $(basename $clipdir)-viz.csv $( cat $clipdir/dates.txt) $( cat $clipdir/bbox.txt)
done
