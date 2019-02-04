# try the folloqing as input - area around NY region
# python DownloadRNNData.py lat_min lon_min lat_max lon_max sdate edate out_path
import sys
import os
import ee
import time
import datetime
import urllib
import zipfile
import glob
import ogr,osr
ee.Initialize()

box_file = sys.argv[1]
sdate = sys.argv[2]
edate = sys.argv[3]
out_base = sys.argv[4]
prefix = sys.argv[5]
basin_file = sys.argv[6]
basin_id = sys.argv[7]
num_boxes = int(sys.argv[8])

# opening the shapefile
print box_file
driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()

bds = driver.Open(basin_file, 0)
bdl = bds.GetLayer()
bdl.SetAttributeFilter("catNum = " + basin_id)
print 'Number of selected sub-basins: ' + str(bdl.GetFeatureCount())


for bfeature in bdl:
    bgeom = bfeature.GetGeometryRef()

cdl.SetSpatialFilter(bgeom)
print 'Number of boxes in the sub-basin: ' + str(cdl.GetFeatureCount())


ctr = 0
for feature in cdl:

    # get boundary information of the partition
    geom = feature.GetGeometryRef()
    #geom.Transform(coordTrans)
    minX, maxX, minY, maxY = geom.GetEnvelope() #bounding box of the box
    curID = feature.GetField('BID')
    lat_min = minY
    lat_max = maxY
    lon_min = minX
    lon_max = maxX
    print lat_min,lon_min,lat_max,lon_max
    cur_box = ee.Geometry.Rectangle(lon_min, lat_min, lon_max, lat_max)
    out_path = out_base + prefix + '-' + str(curID) + '/'
    if os.path.isdir(out_path)==False:
        os.mkdir(out_path)
    print out_path       
    senCol = ee.ImageCollection('COPERNICUS/S2').filterBounds(cur_box).filterDate(sdate,edate)
    senInfo = senCol.getInfo()
    num_images = len(senInfo['features'])
    temp = str([[lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min]])

    i = 0
    flag = 0
    while i<num_images:
        cur_id = senInfo['features'][i]['id'];
        #print("Processing " + cur_id)
        image_id = cur_id[14:]
        data_image = ee.Image(cur_id)
        if os.path.isfile(out_path + image_id + '.zip')==True:
            i = i + 1
            continue

        path = data_image.getDownloadUrl({'name': 'S2B_IW_GRDH_1SDV_' + image_id,'region': temp,'scale':10})
        try:
            urllib.urlretrieve(path, out_path + image_id + '.zip')
            os.system('unzip -q ' + out_path + image_id + '.zip -d ' + out_path)

        except:
            flag = flag+1
            if flag<10:
                print 'Something happened trying again'
                continue
            else:
                print 'Cannot download...moving on'

        i = i + 1
        flag = 0
                
    '''
    senCol = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA').filterBounds(cur_box).filterDate(sdate,edate)
    senInfo = senCol.getInfo()
    num_images = len(senInfo['features'])
    temp = str([[lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min]])
    i = 0
    flag = 0
    while i < num_images:
        cur_id = senInfo['features'][i]['id'];
        print("Processing " + cur_id)
        image_id = cur_id[27:]
        data_image = ee.Image(cur_id)
        if os.path.isfile(out_path + image_id + '.zip')==True:
            i = i + 1
            continue

        path = data_image.getDownloadUrl({'name': image_id,'region': temp,'scale':10})
        try:
            urllib.urlretrieve(path, out_path + image_id + '.zip')

        except:
            flag = flag+1
            if flag<10:
                print 'Something happened trying again'
                continue
            else:
                print 'Cannot download...moving on'

        i = i + 1
        flag = 0
    '''
    
    label_id = 'jrc'
    if os.path.isfile(out_path + label_id + '.zip')==False:

        label_image = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('recurrence')
        path = label_image.getDownloadUrl({'name': label_id,'region': temp,'scale':10})
        flag = 0
        while True:
            try:
                urllib.urlretrieve(path, out_path + label_id + '.zip')
                os.system('unzip -q ' + out_path + label_id + '.zip -d ' + out_path)
            except:
                flag = flag+1
                if flag<10:
                    print 'Something happened trying again'
                    continue
                else:
                    print 'Cannot download...moving on'

            break
    '''

    senCol = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(cur_box).filterDate(sdate,edate)
    senInfo = senCol.getInfo()
    num_images = len(senInfo['features'])
    temp = str([[lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min]])
    print(num_images)
    i = 0
    flag = 0
    while i<num_images:
        cur_id = senInfo['features'][i]['id'];
        #print("Processing " + cur_id)
        image_id = cur_id[18:]
        data_image = ee.Image(cur_id)
        if os.path.isfile(out_path + image_id + '.zip')==True:
            i = i + 1
            continue

        path = data_image.getDownloadUrl({'name': image_id,'region': temp,'scale':10})
        #try:
        urllib.urlretrieve(path, out_path + image_id + '.zip')
        fid = open(out_path + 'meta_' + image_id + '.txt','w')
        pnames = data_image.propertyNames().getInfo()
        #print pnames
        for j in range(0,len(pnames)):
            fid.write(pnames[j] + ',' + str(data_image.get(pnames[j]).getInfo()) + '\n')
        fid.close()
        os.system('unzip -q ' + out_path + image_id + '.zip -d ' + out_path)



        #except:
        #flag = flag+1
        #if flag<10:
        #    print('Something happened trying again')
        #    continue
        #else:
        #    print('Cannot download...moving on')

        i = i + 1
        flag = 0
    
    ctr = ctr + 1
    if ctr==num_boxes:
        break

   

