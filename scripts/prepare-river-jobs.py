import sys
import os
import time
import glob
import ogr,osr
import numpy as np


driver = ogr.GetDriverByName("ESRI Shapefile")
buf_dist = 0.005


# # downloading the world basin file
# os.system('wget http://umnlcc.cs.umn.edu/tmp/hybas_world_lev05_v1c.shp')
# os.system('wget http://umnlcc.cs.umn.edu/tmp/hybas_world_lev05_v1c.shx')
# os.system('wget http://umnlcc.cs.umn.edu/tmp/hybas_world_lev05_v1c.prj')
# os.system('wget http://umnlcc.cs.umn.edu/tmp/hybas_world_lev05_v1c.dbf')


base_dir = sys.argv[1]
prefix = 'cord'
numsamples = sys.argv[2]
input_str = sys.argv[3]

folder_name = base_dir + 'Boxes/'
if os.path.isdir(folder_name)==True:
	os.system('rm -rf ' + folder_name)
os.mkdir(folder_name)


isbasin = 0
islist = 0
ispoint = 0
ispoly = 0
if ',' not in input_str:
    isbasin = 1
    # prefix = prefix + '-' + input_str
if '(' in input_str:
    ispoly = 1
if ',' in input_str and ispoly==0:
    islist = 1

if isbasin==1:
    # removing files if previously downloaded
    numsamples = int(numsamples)
    os.system('rm -f ' + input_str + '.shp*')
    os.system('rm -f ' + input_str + '.shx*')
    os.system('rm -f ' + input_str + '.prj*')
    os.system('rm -f ' + input_str + '.dbf*')

    # downloading point files for the input basin
    os.system('wget --quiet http://umnlcc.cs.umn.edu/tmp/GRWL_vector_V01.01_basins/' + input_str + '.shp')
    os.system('wget --quiet http://umnlcc.cs.umn.edu/tmp/GRWL_vector_V01.01_basins/' + input_str + '.shx')
    os.system('wget --quiet http://umnlcc.cs.umn.edu/tmp/GRWL_vector_V01.01_basins/' + input_str + '.prj')
    os.system('wget --quiet http://umnlcc.cs.umn.edu/tmp/GRWL_vector_V01.01_basins/' + input_str + '.dbf')

    # select random set of points
    points_file = input_str + '.shp'
    bds = driver.Open(points_file, 0)
    bdl = bds.GetLayer()
    pcount = bdl.GetFeatureCount()
    prange = np.arange(pcount)
    np.random.shuffle(prange)
    prange = prange[0:min([pcount,numsamples])]

    # creating the coordinate text file for each point
    for bfeature in bdl:
        if np.sum(prange==int(bfeature.GetField('RPOINT_ID').split('-')[-1]))==0:
            continue
        bgeom = bfeature.GetGeometryRef()
        temp = bgeom.ExportToWkt()
        cord_str = temp.split('(')[1]
        cord_str = cord_str.split(')')[0]
        lon,lat = cord_str.split(' ')
        cord_str = lat + ',' + lon
        fid = open(folder_name + prefix + '-' + bfeature.GetField('RPOINT_ID') + '.txt','w')
        fid.write(cord_str)
        fid.close()

    bds = None
    bdl = None
    os.system('rm -f ' + input_str + '.shp*')
    os.system('rm -f ' + input_str + '.shx*')
    os.system('rm -f ' + input_str + '.prj*')
    os.system('rm -f ' + input_str + '.dbf*')


if islist==1:
	pieces = input_str.split(';')
	ids = numsamples.split(',')
	ctr = 0
	for j in range(0,len(pieces)):
		cord_str = pieces[j]
		id = ids[j]
		fid = open(folder_name + prefix + '-' + id + '.txt','w')
		fid.write(cord_str)
		fid.close()
		ctr = ctr + 1
