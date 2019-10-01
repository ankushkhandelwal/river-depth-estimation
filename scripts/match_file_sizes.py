import os
import sys
import gdal
import glob
import numpy as np

data_dir = sys.argv[1]
img_dir = sys.argv[2]
cinfo = sys.argv[3]
rasterFormat = 'GTiff' # for now assuming output format is going to GTiff
rasterDriver = gdal.GetDriverByName(rasterFormat)

if '/' in cinfo:
    prefix = cinfo[0:-4]
    prefix = prefix[prefix.rfind('/')+1:]
else:
    prefix = cinfo
boxid = prefix

data_dir = data_dir + prefix + '/'
img_dir = img_dir + prefix + '/'

# selecting all classification files
imgs = glob.glob(img_dir + 'S2*.tif')
sizes = {}
for img in imgs:
    print(img)
    ds = gdal.Open(img,0)

    # vv = np.array(ds.GetRasterBand(1).ReadAsArray())
    size_str = str(int(ds.RasterXSize)) + ' ' + str(int(ds.RasterYSize))
    if size_str in sizes:
        sizes[size_str] +=1
    else:
        sizes[size_str] = 1

    ds = None

max_value = 0
for key, value in sizes.items():
        if value>max_value:
            max_value = value
            max_size = key
        print ("% s : % d"%(key, value))
print(max_size)

for img in imgs:
    imgname = img.split('/')[-1]
    imgdir = img[0:img.rfind('/')+1]
    inpimage = imgdir + 'bk/' + imgname
    outimage = img
    os.system('gdalwarp -overwrite -ts ' + max_size + ' ' + inpimage + ' ' + outimage)
# print(np.unique(sizes[:,0]))
# print(np.unique(sizes[:,1]))
