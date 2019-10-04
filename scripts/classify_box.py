import pandas as pd
import numpy as np
import random
import multiprocessing as mp
import gdal
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.initializers
import models
import matplotlib.pyplot as plt
import sys
import os
import glob
import model_library

data_dir = sys.argv[1] # path of box folder where clipped files are stored
img_dir = sys.argv[2]
model_name = sys.argv[3]
cinfo = sys.argv[4]

if '/' in cinfo:
    prefix = cinfo[0:-4]
    prefix = prefix[prefix.rfind('/')+1:]
else:
    prefix = cinfo
boxid = prefix



s2model=models.img_label(pretrained_weights=model_name,type='init')

def pred_fun2(fpath,img_dir,idn):

    # print(idn)
    fname = fpath[fpath.rfind('/')+1:]
    print(fname)
    boxid = fpath.split('/')[-2]
    # print(boxid)
    # if os.path.isdir(img_dir + boxid)==False:
    #     os.mkdir(img_dir + box_id)
    # print(fname)
    if fname[0:2]=='S2':
        ds = gdal.Open(fpath)
        delv = np.array(ds.GetRasterBand(1).ReadAsArray())

        brows = delv.shape[0]
        bcols = delv.shape[1]
        # print(brows,bcols)


        all_bands=np.zeros((brows,bcols,13),float)
        b_arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
        for b in range(all_bands.shape[2]):
            all_bands[:,:,b]= np.array(ds.GetRasterBand(int(b_arr[b])).ReadAsArray())

        cloud_matrix = model_library.cloud_pred(all_bands)

        pred_bands_org=all_bands[:,:,[1,2,3,4,5,6,7,11,12]]
        min_arr = np.zeros((pred_bands_org.shape[2],))-1
        max_arr = np.zeros((pred_bands_org.shape[2],))-1
        mean_arr = np.zeros((pred_bands_org.shape[2],))-1
        for i in range(pred_bands_org.shape[2]):
            b_max=np.max(pred_bands_org[:,:,i])
            b_min=np.min(pred_bands_org[:,:,i])
            min_arr[i] = b_min
            max_arr[i] = b_max
            mean_arr[i] = np.mean(pred_bands_org[:,:,i])

        if np.sum(min_arr>0)==0 and np.sum(max_arr>0)==0:
            pred_labels=np.zeros((brows,bcols),float)
            pred_labels[:,:] = 3
            np.save(img_dir + boxid + '/C' + fname[0:-4],pred_labels)
            print('empty image')
            return

        for i in range(pred_bands_org.shape[2]):
            b_min = min_arr[i]
            b_max = max_arr[i]
            # b_mean =
            # pred_bands_org[:,:,i]=(pred_bands_org[:,:,i]-b_min)/(b_max-b_min)

        if brows<96:
            temp = pred_bands_org.copy()
            ctemp = cloud_matrix.copy()
            nof = pred_bands_org.shape[2]

            pred_bands_org = np.zeros((96,bcols,nof))
            pad_ind = int((96-brows)/2)
            for i in range(nof):
                pred_bands_org[pad_ind:pad_ind+brows,:,i] = temp[:,:,i]

            cloud_matrix = np.zeros((96,bcols,2))+2
            pad_ind = int((96-brows)/2)
            for i in range(2):
                cloud_matrix[pad_ind:pad_ind+brows,:,i] = ctemp[:,:,i]

            brows = 96

        if bcols<96:

            temp = pred_bands_org.copy()
            ctemp = cloud_matrix.copy()
            nof = pred_bands_org.shape[2]

            pred_bands_org = np.zeros((brows,96,nof))
            pad_ind = int((96-bcols)/2)
            for i in range(nof):
                pred_bands_org[:,pad_ind:pad_ind+bcols,i] = temp[:,:,i]

            cloud_matrix = np.zeros((brows,96,2))
            pad_ind = int((96-bcols)/2)
            for i in range(2):
                cloud_matrix[:,pad_ind:pad_ind+bcols,i] = ctemp[:,:,i]
            bcols = 96

        # print(brows,bcols)
        step = 48
        rarr = np.arange(0,brows,step)
        rarr = rarr[0:-1*int(96/step)]
        rarr = np.append(rarr,brows-96)
        carr = np.arange(0,bcols,step)
        carr = carr[0:-int(96/step)]
        carr = np.append(carr,bcols-96)
        # print(rarr,carr)

        pred_labels=np.zeros((brows,bcols),float)
        pred_deno=np.zeros((brows,bcols),float)
        ppad = 24
        # print('ready for prediction')
        # print(pred_bands_org.shape)
        X = np.zeros((len(rarr)*len(carr),96,96,9))
        ctr = 0
        for r in rarr:
            for c in carr:
                # print(r,c,ctr)
                # print(ctr)
                pred_bands=pred_bands_org[r:r+96,c:c+96,:].copy()
                min_arr = np.zeros((pred_bands.shape[2],))-1
                max_arr = np.zeros((pred_bands.shape[2],))-1
                for i in range(pred_bands.shape[2]):
                    b_max=np.max(pred_bands[:,:,i])
                    b_min=np.min(pred_bands[:,:,i])
                    min_arr[i] = b_min
                    max_arr[i] = b_max
                for i in range(pred_bands.shape[2]):
                    b_min = min_arr[i]
                    b_max = max_arr[i]
                    # pred_bands[:,:,i]=(pred_bands[:,:,i]-b_min)/(b_max-b_min)
                    pred_bands[:,:,i]=pred_bands[:,:,i]*1.0/mean_arr[i]
                X[ctr,:,:,:] = pred_bands
                ctr = ctr + 1
        Y=s2model.predict(X)
        # print(Y.shape)
        ctr = 0
        for r in rarr:
            for c in carr:
                CNN_pred_matrix = np.squeeze(Y[ctr,:,:,:])
                pred_labels[r+ppad:r+96-ppad,c+ppad:c+96-ppad]=pred_labels[r+ppad:r+96-ppad,c+ppad:c+96-ppad]+CNN_pred_matrix[ppad:96-ppad,ppad:96-ppad]
                # print(5)
                pred_deno[r+ppad:r+96-ppad,c+ppad:c+96-ppad]=pred_deno[r+ppad:r+96-ppad,c+ppad:c+96-ppad] + 1
                ctr = ctr + 1

        pad_inds = np.logical_or(pred_deno==0,np.sum(pred_bands_org>0,axis=2)==0)
        # print(8)
        pred_deno[pred_deno==0] = 1
        pred_labels = pred_labels*1.0/pred_deno
        pred_labels[pad_inds] = 3
        pred_labels_cloud = pred_labels.copy()
        pred_labels_cloud[cloud_matrix[:,:,1]==1] = 2
        # plt.figure()
        # plt.imshow(pred_labels)
        # plt.show()

        # D = np.sum(pred_labels_cloud==2)
        # print(9)
        np.save(img_dir + boxid + '/M' + fname[0:-4],pred_labels_cloud)
        np.save(img_dir + boxid + '/C' + fname[0:-4],pred_labels)
        return
        # print(pred_bands_org.shape,pred_labels.shape)
        # plot_pred(pred_bands_org.copy(),np.expand_dims(pred_labels,axis=2),img_dir + fname[0:-4] + '.png')
        # plot_pred_cloud(pred_bands_org.copy(),np.expand_dims(pred_labels,axis=2),np.expand_dims(pred_labels_cloud,axis=2),img_dir + str(int(D)) + '_' + boxid + '_' + fname[0:-4] + '.png')
        # model_library.cloud_pred(full_band_matrix)

print(data_dir)
flist = glob.glob(data_dir + prefix + '/S2*.tif')
print('Number of files: ' + str(len(flist)))
tasks = []
for fpath in flist:
    # if '20151218' not in fpath:
        # continue
    fname = fpath[fpath.rfind('/')+1:]
    boxid = fpath.split('/')[-2]
    if os.path.isdir(img_dir + boxid)==False:
        os.mkdir(img_dir + boxid)
    print(img_dir)
    if os.path.isfile(img_dir + boxid + '/C' + fname[0:-4] + '.npy')==False:
        pred_fun2(fpath,img_dir,1)
