import sys
import os
import gdal
import tensorflow
import keras
import skimage
import glob


base_dir = sys.argv[1]
prefix = sys.argv[2]
numsamples = int(sys.argv[3])

for i in range(numsamples):
	folder_name = base_dir + prefix + '-' + str(i) + '/'
	os.mkdir(folder_name)
	fid = open(folder_name + 'test.txt','w')
	fid.write('this is a test line')
	fid.close()
sys.exit()

