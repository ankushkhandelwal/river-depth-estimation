export PATH=/home/kumarv/khand035/.conda/envs/tf_cpu_clone_jup/bin:/home/kumarv/khand035/.conda/envs/gdalenv/bin/:/home/kumarv/khand035/Projects/CodeLibrary/pycharm-community-2018.2.5/bin:$PATH
export LD_LIBRARY_PATH=/home/kumarv/khand035/.conda/envs/tf_cpu_clone_jup/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/kumarv/khand035/.conda/envs/gdalenv/lib/python2.7/site-packages/
export KERAS_BACKEND=tensorflow

#python GEEScript.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$1 $2 $3 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/WBD_Delineation_wgs84.shp 100 1
#python GEEScript-single.py $1 $2 $3 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4
#python GEEClipDEM.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$1 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/dem_wgs84_sqcut.tif credentials.txt /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/WBD_Delineation_wgs84.shp 100 1
for clipdir in /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$4*/ ; 
do
python ProcessZip_wrapper.py $clipdir $(basename $clipdir).txt 1
python ExtractRiver.py $clipdir $(basename $clipdir).txt 1
 
done
