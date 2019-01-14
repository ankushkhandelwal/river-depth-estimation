import os
import sys
import numpy as np
import pandas
import matplotlib.pyplot as plt
sys.path.append('/home/kumarv/khand035/.conda/envs/gdalenv/lib/python2.7/site-packages/')
from osgeo import gdal,ogr
import glob
from PIL import Image
import skimage.feature

#data_dir = '/home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/BIDS2100-53/'
data_dir = sys.argv[1]

print 'Removing cloud contanimated maps...'

# creating directory to store cloudy images
if os.path.isdir(data_dir + 'badmaps/')==False:
    os.mkdir(data_dir + 'badmaps/')


# selecting all merged maps
imgs = glob.glob(data_dir + 'F*' + '*.tif')
for img in imgs:
    ds = Image.open(img)
    ds_arr = np.array(ds)
    # moving the map if it contains even 1 invalid pixel
    if np.sum(ds_arr>1)>0:
                os.system('mv ' + img + ' ' + data_dir + 'badmaps/')