import os
import gdal
import numpy as np
#import matplotlib.pyplot as plt
import glob

def run(fname,data_dir,ext):
    rasterFormat = 'GTiff'  # for now assuming output format is going to GTiff
    rasterDriver = gdal.GetDriverByName(rasterFormat)
    vv = -1
    vh = -1
    for filename in glob.glob(data_dir + '*' + fname + '*.'+ ext):
        
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

    vv_threshold = get_threshold(vv_flat)
    vh_threshold = get_threshold(vh_flat)
    #print('VV Threshold:', vv_threshold)
    #print('VH Threshold:', vh_threshold)


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
    mds = rasterDriver.Create(data_dir + '/' + fname + '.' + ext, full_xsize, full_ysize, 1, gdal.GDT_Byte)
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

