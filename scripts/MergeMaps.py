import os 
import sys
import gdal
import glob
import numpy as np
data_dir = sys.argv[1]


# selecting all classification files
imgs = glob.glob(data_dir + 'M*' + '*.tif')
ext = 'tif'
rasterFormat = 'GTiff'  # for now assuming output format is going to GTiff
rasterDriver = gdal.GetDriverByName(rasterFormat)
fnames = []
print 'Merging classification maps...'
# extracting date strings
for img in imgs:
    tname = img[img.rfind('/')+1:]
    #print tname
    #print tname[3:34]
    fnames.append(tname[3:11])

# sorting dates
fnames = list(set(fnames))
fnames.sort()
#print fnames

# merging classification maps for each date
for i in range(0,len(fnames)):
    fname = fnames[i]
    imgs = glob.glob(data_dir + 'M*' + fname + '*.tif')
    #print fname
    #print len(imgs)
    
    # storing sensor type and valid pixel count for each map
    stype = np.zeros((len(imgs),))
    gcnt = np.zeros((len(imgs),))
    for j in range(0,len(imgs)):
        img = imgs[j]
        tname = img[img.rfind('/')+1:]
        stype[j] = 3-int(tname[1])
        ds = gdal.Open(img)
        ds_band = ds.GetRasterBand(1)
        numchannels = ds.RasterCount
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        full_xsize = ds_band.XSize
        full_ysize = ds_band.YSize
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        mask_arr = np.array(ds_band.ReadAsArray())
        gcnt[j] = np.sum(mask_arr>1)
    
    
    # giving first priority to Sentinel-2 maps
    # then number of valid pixels
    sinds = np.lexsort((gcnt,stype))
    
    # merging classification maps in the order
    fmap = np.zeros((rows,cols))+4
    for j in range(0,len(imgs)):
        img = imgs[sinds[j]]
        #print img
        ds = gdal.Open(img)
        mask_arr = np.array(ds.GetRasterBand(1).ReadAsArray())
        new_inds = np.logical_and(fmap>1,mask_arr<=1)
        fmap[new_inds] = mask_arr[new_inds] 
        ds = None
    
    # Saving the merged map
    #print data_dir + '/F3_' + fname + '.' + ext
    mds = rasterDriver.Create(data_dir + '/F3_' + fname + '.' + ext, full_xsize, full_ysize, 1, gdal.GDT_Byte)
    mds.SetGeoTransform(geotransform)
    mds.SetProjection(projection)
    mds.GetRasterBand(1).WriteArray(fmap)
    mds = None
