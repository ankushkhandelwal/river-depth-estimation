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
folder_name = base_dir + 'Boxes/'
if os.path.isdir(folder_name)==True:
	os.system('rm -rf ' + folder_name)
os.mkdir(folder_name)
for i in range(numsamples):
	fid = open(folder_name + prefix + '-' + str(i) + '.txt','w')
	fid.write(str(i)+ ',' + str(i+1) + '\n')
	fid.close()
sys.exit()
