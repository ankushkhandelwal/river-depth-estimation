import os
import sys
import gdal,ogr

box_file = '/home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/GageBoxes2_wbd.shp'
data_dir = '/home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/'
driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()
fid = open('GEECalls.sh','w')
catNums = []
for feature in cdl:
    BID = feature.GetField('BID')
    catNum = feature.GetField('catNum')
    cur_dir = data_dir + 'BIDS2' + str(catNum) + '-' + str(BID)
    #if os.path.isdir(cur_dir)==True:
    #    continue
    catNums.append(catNum)

catNums = list(set(catNums))
for catNum in catNums:
    fid.write('sh main-wrapper.sh ' + str(catNum) + ' 2016-01-01 2017-03-31 BIDS2' + str(catNum) + ' -1\n')
    
fid.close()
