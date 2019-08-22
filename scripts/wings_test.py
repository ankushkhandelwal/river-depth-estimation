import sys
import os
import gdal
import tensorflow
import keras
import skimage
import glob


folder_name = sys.argv[1]
os.mkdir(folder_name)
fid = open(folder_name + 'test.txt','w')
fid.write('this is a test line')
fid.close()
sys.exit()

