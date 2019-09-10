import os
import glob
import numpy as np
import gdal
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import sys

def ORBITE(pred_labels,rank_labels):
    pred_labels[rank_labels==3] = 3
    rows,cols = pred_labels.shape
    rank_labels = rank_labels.flatten()
    pred_labels = pred_labels.flatten()
    rank_labels[rank_labels==3] = 0
    rank_labels[rank_labels==2] = 0.5
    pred_labels[pred_labels==3] = 0
    pred_labels[pred_labels==2] = 0.5

    ix = np.argsort(rank_labels)
    # plt.figure()
    # plt.plot(pred_labels[ix])
    # plt.show()
    ord_labels = pred_labels[ix]
    ord_labels = np.append(0,ord_labels)
    sum1 = np.cumsum(ord_labels)
    sum2 = np.cumsum(1 - ord_labels)
    score = sum1 + np.sum(sum2) - sum2
    ind = np.argmin(score)
    new_labels = ord_labels.copy()
    new_labels[:] = 0
    new_labels[0:ind] = 0
    new_labels[ind:] = 1
    new_labels = new_labels[1:]
    flabels = pred_labels.copy()
    flabels[ix] = new_labels
    # f,(ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.imshow(np.reshape(rank_labels,(rows,cols)))
    # ax2.imshow(np.reshape(pred_labels,(rows,cols)))
    # ax3.imshow(np.reshape(flabels,(rows,cols)))
    # # ax4.imshow(np.reshape(pred_labels[ix],(rows,cols)))
    # plt.show()
    flabels = np.reshape(flabels,(rows,cols))
    return flabels




def ProcessImage(cpath,pred_summary,fig_name,idn):
    cname = cpath[cpath.rfind('/')+1:]
    # if os.path.isfile(img_dir + boxid + '/C' + cname[0:-4] + '.png')==True:
    #     continue
    ds = gdal.Open(cpath)

    delv = np.array(ds.GetRasterBand(1).ReadAsArray())
    brows = delv.shape[0]
    bcols = delv.shape[1]
    if brows!=pred_summary.shape[0] or bcols!=pred_summary.shape[1]:
        return

    # print(brows,bcols)
    all_bands=np.zeros((brows,bcols,13),float)
    b_arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
    for b in range(all_bands.shape[2]):
        all_bands[:,:,b]= np.array(ds.GetRasterBand(int(b_arr[b])).ReadAsArray())

    pred_bands_org=all_bands[:,:,[1,2,3,4,5,6,7,11,12]]
    min_arr = np.zeros((pred_bands_org.shape[2],))-1
    max_arr = np.zeros((pred_bands_org.shape[2],))-1
    for i in range(pred_bands_org.shape[2]):
        b_max=np.max(pred_bands_org[:,:,i])
        b_min=np.min(pred_bands_org[:,:,i])
        min_arr[i] = b_min
        max_arr[i] = b_max

    if np.sum(min_arr>0)==0 and np.sum(max_arr>0)==0:
        return

    for i in range(pred_bands_org.shape[2]):
        b_min = min_arr[i]
        b_max = max_arr[i]
        pred_bands_org[:,:,i]=(pred_bands_org[:,:,i]-b_min)/(b_max-b_min)
    pred_labels = np.load(img_dir + boxid + '/C' + cname[0:-4] + '.npy')
    pred_labels_cloud = np.load(img_dir + boxid + '/M' + cname[0:-4] + '.npy')
    pred_labels_cloud_org = pred_labels_cloud.copy()
    if np.sum(np.logical_and(pred_labels_cloud==2,pred_summary>0))*1.0/np.sum(np.logical_and(pred_labels_cloud<=2,pred_summary>0))<0.5:
        # print('applying orbit E')
        pred_labels_cloud = ORBITE(pred_labels_cloud.copy(),pred_summary.copy())

    np.save(img_dir + boxid + '/O' + cname[0:-4],pred_labels_cloud)

    return



data_dir = sys.argv[1]
img_dir = data_dir
fig_dir = data_dir
cinfo = sys.argv[2]
prefix = cinfo[0:-4]
if '/' in prefix:
    prefix = prefix[prefix.rfind('/')+1:]
boxid = prefix

ctr = 0
if len(glob.glob(img_dir  + boxid + '/M*.npy'))>0:
    for cpath in glob.glob(img_dir  + boxid + '/M*.npy'):
        pred_labels = np.load(cpath)
        trows,tcols = pred_labels.shape
        if ctr == 0:
            bad_labels = pred_labels>1
            pred_summary = pred_labels
            pred_summary[bad_labels] = 0
            pred_deno = bad_labels==0
            pred_summary = pred_summary.astype(float)
            pred_deno = pred_deno.astype(float)
            ctr = 1
        elif trows==pred_summary.shape[0] and tcols==pred_summary.shape[1]:
            bad_labels = pred_labels>1
            pred_summary[bad_labels==0] = pred_summary[bad_labels==0] + pred_labels[bad_labels==0]
            pred_deno[bad_labels==0] = pred_deno[bad_labels==0] + 1


    bad_labels = pred_deno==0
    pred_deno[pred_deno==0] = 1
    pred_summary = pred_summary/pred_deno
    pred_summary[bad_labels] = 3
    print_count = 0
    tasks = []
    for cpath in glob.glob(img_dir  + boxid + '/S2*.tif'):
        cname = cpath[cpath.rfind('/')+1:]
        cdate = cname.split('__')[-1]
        cdate = cdate[0:8]
        fig_name = fig_dir + boxid + '/' + cdate + '_C' + cname[0:-4] + '.png'
        print(cname)
        ProcessImage(cpath,pred_summary.copy(),fig_name,1)
