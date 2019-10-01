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
sdate = sys.argv[1]
edate = sys.argv[2]
out_base = sys.argv[3]
cinfo = sys.argv[4]
buf_dist = float(sys.argv[5])
prefix = cinfo[0:-4]
if '/' in prefix:
    prefix = prefix[prefix.rfind('/')+1:]

with open(cinfo) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
inpID = content[0]
print(inpID)

temp = inpID.split(',')
lat = float(temp[0])
lon = float(temp[1])

point = ogr.Geometry(ogr.wkbPoint)
point.AddPoint(lon,lat)
box = point.Buffer(buf_dist)
minX, maxX, minY, maxY = box.GetEnvelope() #bounding box of the box

lat_min = minY
lat_max = maxY
lon_min = minX
lon_max = maxX

print(lat_min,lon_min,lat_max,lon_max)

lat_min = float(lat_min)
lon_min = float(lon_min)
lat_max = float(lat_max)
lon_max = float(lon_max)
# print lat_min,lon_min,lat_max,lon_max
cur_box = ee.Geometry.Rectangle(lon_min, lat_min, lon_max, lat_max)
out_path = out_base + prefix + '/'
if os.path.isdir(out_path)==False:
    os.mkdir(out_path)

senCol = ee.ImageCollection('COPERNICUS/S2').filterBounds(cur_box).filterDate(sdate,edate)
senInfo = senCol.getInfo()
num_images = len(senInfo['features'])
temp = str([[lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min]])

i = 0
flag = 0
while i<num_images:
    cur_id = senInfo['features'][i]['id'];
    # if '20180620T081859' not in cur_id:
        # i = i + 1
        # continue
    print("Processing " + cur_id)
    image_id = cur_id[14:]
    data_image = ee.Image(cur_id)
    # if os.path.isfile(out_path + image_id + '.zip')==True:
        # i = i + 1
        # continue

    path = data_image.getDownloadUrl({'name': 'S2B_IW_GRDH_1SDV_' + image_id,'region': temp,'scale':10})
    try:
        urllib.urlretrieve(path, out_path + image_id + '.zip')
        # os.system('unzip -q ' + out_path + image_id + '.zip -d ' + out_path)

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
'''
year = sdate[0:4]
label_id = 'CDL_' + year
if os.path.isfile(out_path + label_id + '.zip')==False:

    label_image = ee.Image('USDA/NASS/CDL/' + year)
    path = label_image.getDownloadUrl({'name': label_id,'region': temp,'scale':10})
    flag = 0
    while True:
        try:
            urllib.urlretrieve(path, out_path + label_id + '.zip')

        except:
            flag = flag+1
            if flag<10:
                print 'Something happened trying again'
                continue
            else:
                print 'Cannot download...moving on'

        break
'''
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
    print("Processing " + cur_id)
    image_id = cur_id[18:]
    data_image = ee.Image(cur_id)
    #if os.path.isfile(out_path + image_id + '.zip')==True:
    #    i = i + 1
    #    continue

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
'''
