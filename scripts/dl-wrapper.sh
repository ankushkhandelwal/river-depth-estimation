python DLScript.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$1 $2 $3 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4
python GEEClipDEM.py /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$1 /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/dem_wgs84_sqcut.tif credentials.txt /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/ $4
for clipdir in /home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/$4*/ ; 
do
python ProcessZip_wrapper.py $clipdir $(basename $clipdir).txt 2 
done
