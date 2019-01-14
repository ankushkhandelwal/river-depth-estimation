import os
import gdal
import numpy as np
#import matplotlib.pyplot as plt
import glob
from skimage import filters

# function for classification of Sentinel-2 images
def runS2(fname,data_dir,ext):
    rasterFormat = 'GTiff'  # for now assuming output format is going to GTiff
    rasterDriver = gdal.GetDriverByName(rasterFormat)
    s2bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20', 'QA60']
    green = s2bands.index('B3')
    blue = s2bands.index('B2')
    nir = s2bands.index('B8')
    swir1 = s2bands.index('B11')
    swir2 = s2bands.index('B12')
    cmask = s2bands.index('QA60')
    
    imgs = glob.glob(data_dir + 'S2*' + fname + '*.tif')
    
    if len(imgs)>0:
        # extracting unique scene ids
        scene_arr = []
        for img in imgs:
            tname = img[img.rfind('/')+1:]
            scene_arr.append(tname[49:55])
        scene_arr = list(set(scene_arr))
        #print scene_arr
        #for scene in scene_arr:
        #    print scene
        # classifying data corresponding to each scene
        for scene in scene_arr:
            #print scene
            imgs = glob.glob(data_dir + 'S2*' + fname + '*' + scene + '*.tif')
            #imgs.sort()
            #print imgs
            if len(imgs)==16:
                
                for j in range(0,len(imgs)):
                    filename = imgs[j]
                    #print filename
                    ds = gdal.Open(filename,0)
                    ds_band = ds.GetRasterBand(1)
                    geotransform = ds.GetGeoTransform()
                    projection = ds.GetProjection()
                    full_xsize = ds_band.XSize
                    full_ysize = ds_band.YSize
                    ds_arr = np.array(ds_band.ReadAsArray())
                    rows,cols = ds_arr.shape
                    if j==0:
                        data = np.zeros((rows,cols,len(s2bands)))
                    data[:,:,j] = ds_arr

                ndwi_num = (data[:,:,green] - data[:,:,nir])*1.0
                ndwi_den = (data[:,:,green] + data[:,:,nir])*1.0
                ndwi = np.divide(ndwi_num,ndwi_den)
                #wth = filters.threshold_otsu(ndwi)
                #print wth
                #f, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True,figsize=(10,5))
                #ax1.imshow(ndwi)
                #ax2.imshow(ndwi>wth)
                #plt.show()
                # currently using static threshold
                water_mask = ndwi>-0.11
                water_mask = water_mask.astype('int')
                water_mask[data[:,:,cmask]==1024] = 2 # thin clouds
                water_mask[data[:,:,cmask]==2048] = 3 # thick clouds

                mds = rasterDriver.Create(data_dir + '/M2_' + fname + '_' + scene + '.' + ext, full_xsize, full_ysize, 1, gdal.GDT_Byte)
                mds.SetGeoTransform(geotransform)
                mds.SetProjection(projection)
                mds.GetRasterBand(1).WriteArray(water_mask)
                mds = None
                ds = None
                #return np.sum(water_mask)    

            
            else:
                print 'bad data in S2'
                #return -1

    return -1
   
#function for classification of Sentinel-1 images
def runS1(fname,data_dir,ext):
    rasterFormat = 'GTiff'  # for now assuming output format is going to GTiff
    rasterDriver = gdal.GetDriverByName(rasterFormat)
    vv = -1
    vh = -1
    for filename in glob.glob(data_dir + 'S1*' + fname + '*.'+ ext):
        
        #print filename
        if 'vv' in filename or 'VV' in filename:
            dsv = gdal.Open(filename,0)
            vv = dsv.GetRasterBand(1)
        if 'vh' in filename or 'VH' in filename:
            dsh = gdal.Open(filename,0)
            vh = dsh.GetRasterBand(1)
    if vh==-1 or vv==-1:
        return -1

    geotransform = dsv.GetGeoTransform()
    projection = dsv.GetProjection()
    band = dsv.GetRasterBand(1)
    full_xsize = band.XSize
    full_ysize = band.YSize

    vv_arr = np.array(vv.ReadAsArray())
    vh_arr = np.array(vh.ReadAsArray())
    rows,cols = vv_arr.shape
    #print vv_arr.shape

    vv_flat = vv_arr.flatten()
    vh_flat = vh_arr.flatten()
    #n, bins, patches = plt.hist(x=vh_flat, bins='auto', color='#0504aa',
    #                        alpha=0.7, rwidth=0.85)
    #vh_threshold = bins[3]
    
    #n, bins, patches = plt.hist(x=vv_flat, bins='auto', color='#0504aa',
    #                        alpha=0.7, rwidth=0.85)
    n, bins = np.histogram(x=vv_flat, bins='auto')
    vv_threshold = bins[3]
    
    water_mask = vv_arr<vv_threshold
    #vv_threshold = get_threshold(vv_flat)
    #vh_threshold = get_threshold(vh_flat)
    
    #print('VV Threshold:', vv_threshold)
    #print('VH Threshold:', vh_threshold)

    #print('OVV Threshold:', ovv_threshold)
    #print('OVH Threshold:', ovh_threshold)

    #f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, sharex=True, sharey=True,figsize=(10,5))   
    #ax1.imshow(vh_arr)
    #ax2.imshow(vh_arr<vh_threshold)
    #ax3.imshow(vv_arr)
    #ax4.imshow(vv_arr<vv_threshold)
    #plt.show()
    #ax4.imshow(vh_arr)
    #ax5.imshow(vh_arr<vh_threshold)
    #ax6.imshow(vh_arr<ovh_threshold)
    
    '''    
    water_mask = []
    for i,j in zip(vv_arr, vh_arr):
        row = []
        for m,n in zip(i,j):
            if m < vv_threshold and n < vh_threshold:
                row.append(1)
            else:
                row.append(0)
        water_mask.append(row)

    # converting water mask to geotif
    water_mask = np.reshape(water_mask,(rows,cols))
    '''
    
    
    mds = rasterDriver.Create(data_dir + '/M1_' + fname + '.' + ext, full_xsize, full_ysize, 1, gdal.GDT_Byte)
    mds.SetGeoTransform(geotransform)
    mds.SetProjection(projection)
    mds.GetRasterBand(1).WriteArray(water_mask)
    mds = None
    ds = None

    # ToDo: Save tif 
    #print water_mask.shape

    area = np.sum(water_mask)
    #print area
    return area

def get_threshold(arr):
    hist = np.histogram(arr, bins=100)
    counts = hist[0]
    values = hist[1][:100]
    
    total = np.sum(np.array(counts))
    p_sum = np.sum(counts* values)
    p_mean = p_sum/total
    
    best_error = 0
    threshold = 0
    
    for index in range(len(counts)):
        low_count_ar = counts[0:index+1]
        low_count_values = values[0:index+1]
        high_count_ar = counts[index:]
        high_count_values = values[index:]
        
        low_count = np.sum(low_count_ar)
        low_mean = 0
        for i,j in zip(low_count_ar, low_count_values):
            low_mean += i*j
        low_mean /= low_count
            
        high_count = np.sum(high_count_ar)
        high_mean = 0
        for i,j in zip(high_count_ar, high_count_values):
            high_mean += i*j
        high_mean /= high_count
        
        error = low_count*pow(low_mean-p_mean,2) + high_count*pow(high_mean-p_mean,2)
        if error > best_error:
            best_error = error
            threshold = values[index]
        
    return threshold

