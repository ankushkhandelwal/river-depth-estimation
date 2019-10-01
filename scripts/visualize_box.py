import pandas as pd
import numpy as np
import random
import multiprocessing as mp
import gdal
# import models
import matplotlib.pyplot as plt
import sys
import os
import glob
# import model_library
import dproc
import time
import skimage.exposure

data_dir = sys.argv[1] # path of box folder where clipped files are stored
img_dir = sys.argv[2]
# model_name = sys.argv[3]
cinfo = sys.argv[3]
if '/' in cinfo:
    prefix = cinfo[0:-4]
    prefix = prefix[prefix.rfind('/')+1:]
else:
    prefix = cinfo
boxid = prefix
# data_dir = data_dir + prefix + '/'
# img_dir = data_dir

def CreateAreaTimeseries(ilist,img_dir):
    numt = len(ilist)
    area_ts = np.zeros((numt,))-1
    dates_ts = np.zeros((numt,))-1

    for j in range(0,numt):
        ipath = ilist[j]
        iname = ipath.split('/')[-1]
        idir = ipath[0:ipath.rfind('/')+1]
        boxid = ipath.split('/')[-2]
        if os.path.isfile(img_dir + boxid + '/O' + iname[0:-4] + '.npy')==False:
            dates_ts[j] = int(iname[-34:-26])
            area_ts[j] = -1
            continue
        pred_labels_cor = np.load(img_dir + boxid  + '/O' + iname[0:-4] + '.npy')
        area_ts[j] = np.sum(pred_labels_cor==1)
        if np.sum(pred_labels_cor>1)>0:
            area_ts[j] = -1
        print(iname,iname[-34:-26])
        dates_ts[j] = int(iname[-34:-26])

    ix = np.argsort(dates_ts)
    area_ts = area_ts[ix]
    dates_ts = dates_ts[ix]
    return area_ts,dates_ts



def plot_pred_video(X,Y,T,D,savepath):

    iname = savepath.split('/')[-1]
    idir = savepath[0:savepath.rfind('/')+1]
    cur_date = int(iname[-34:-26])
    # print(X.shape,Y.shape)
    rgb_map=np.zeros((X.shape[0],X.shape[1],3),float)
    rgb_map[:,:,0]= X[:,:,0]
    rgb_map[:,:,1]= X[:,:,1]
    rgb_map[:,:,2]= X[:,:,2]
    rgb_map_scaled = skimage.exposure.equalize_hist(rgb_map)
    # for j in range(rgb_map.shape[2]):
    #     temp = rgb_map[:,:,j].copy()
    #     m = np.mean(temp)
    #     s = np.std(temp)
    #     temp = (temp - m)*1.0/s
    #     temp = temp - np.min(temp)
    #     temp = temp*1.0/np.max(temp)
    #     temp = np.flo
    #     rgb_map[:,:,j] = temp
    # rgb_map = rgb_map.astype(int)
    # print(rgb_map.shape)
    # plt.figure()
    # plt.show()
    # f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(5,5))
    # ax1.imshow(rgb_map)
    rgb_map_labels = skimage.exposure.equalize_hist(rgb_map)
    tY = np.squeeze(Y[:,:,:])

    temp = rgb_map_labels[:,:,0]
    temp[tY==1] = 0
    rgb_map_labels[:,:,0] = temp
    temp = rgb_map_labels[:,:,1]
    temp[tY==1] = 0
    rgb_map_labels[:,:,1] = temp
    temp = rgb_map_labels[:,:,2]
    temp[tY==1] = 1
    rgb_map_labels[:,:,2] = temp

    temp = rgb_map_labels[:,:,0]
    temp[tY==2] = 1
    rgb_map_labels[:,:,0] = temp
    temp = rgb_map_labels[:,:,1]
    temp[tY==2] = 1
    rgb_map_labels[:,:,1] = temp
    temp = rgb_map_labels[:,:,2]
    temp[tY==2] = 1
    rgb_map_labels[:,:,2] = temp



    tY = np.squeeze(Y[:,:,:])
    tY[0,0] = 0
    tY[0,1] = 1
    tY[0,2] = 2
    tY[0,3] = 3
    tY[tY>1] = np.nan
    # ax3.imshow(tY)

    fig = plt.figure(figsize=(6, 6))
    ax1= fig.add_subplot(2,2,1)
    ax2= fig.add_subplot(2,2,2)
    ax3= fig.add_subplot(2,1,2)
    ax1.imshow(rgb_map_scaled)
    ax2.imshow(rgb_map_labels)
    T[T==-1] = np.nan
    T[T==0] = np.nan
    T = T*0.0001
    Dstr = []
    for d in D:
        Dstr.append(str(int(d)))

    x = np.where(D==cur_date)[0][0]
    savepath = idir + 'img' + str(int(x)).zfill(3) + '.png'
    ax3.plot(T,'-*')
    ax3.plot([x,x],[np.nanmin(T), np.nanmax(T)],'-r')
    ax3.set_xticks(np.arange(0,len(T),10))
    plt.title(Dstr[x])
    temp = np.arange(0,len(T),10).astype(int)
    ax3.set_xticklabels(Dstr[0:len(T):10])
    plt.xticks(rotation=45)
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    plt.grid()
    # plt.show()

    # return



    # savepath=''
    if savepath=='':
        print('not saving')
        plt.show()
    else:
        fig.savefig(savepath,dpi=300)
    plt.close()
    return

def plot_pred_cloud(X,Y,Z,W,savepath):
    # print(X.shape,Y.shape)
    rgb_map=np.zeros((X.shape[0],X.shape[1],3),float)
    rgb_map[:,:,0]= X[:,:,0]
    rgb_map[:,:,1]= X[:,:,1]
    rgb_map[:,:,2]= X[:,:,2]
    # print(rgb_map.shape)
    # plt.figure()
    # plt.show()
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(5,5))
    ax1.imshow(rgb_map)
    tY = np.squeeze(Y[:,:,:])
    tY[0,0] = 0
    tY[0,1] = 1
    tY[0,2] = 2
    tY[0,3] = 3
    tY[tY>1] = np.nan
    ax3.imshow(tY)

    tZ = np.squeeze(Z[:,:,:])
    tZ[0,0] = 0
    tZ[0,1] = 1
    tZ[0,2] = 2
    tZ[0,3] = 3
    tZ[tZ>1] = np.nan
    ax2.imshow(tZ)


    tW = np.squeeze(W[:,:,:])
    tW[0,0] = 0
    tW[0,1] = 1
    tW[0,2] = 2
    tW[0,3] = 3
    tW[tW>1] = np.nan
    ax4.imshow(tW)
    plt.tight_layout()
    # savepath=''
    if savepath=='':
        print('not saving')
        plt.show()
    else:
        f.savefig(savepath)
    plt.close()
    return


def pred_fun2(fpath,img_dir,idn):

    # print(idn)

    fname = fpath[fpath.rfind('/')+1:]
    print(fname)
    boxid = fpath.split('/')[-2]
    # if os.path.isfile(img_dir + boxid + '/' + fname[0:-4] + '.png')==True:
        # return
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

        start_time = time.time()
        pred_bands_org=np.zeros((brows,bcols,3),float)
        b_arr = np.array([8,4,3])
        for b in range(pred_bands_org.shape[2]):
            pred_bands_org[:,:,b]= np.array(ds.GetRasterBand(int(b_arr[b])).ReadAsArray())
        # print("--- %s seconds at 1---" % (time.time() - start_time))
        # cloud_matrix = dproc.cloud_pred(all_bands)
        # print("--- %s seconds at 2---" % (time.time() - start_time))

        min_arr = np.zeros((pred_bands_org.shape[2],))-1
        max_arr = np.zeros((pred_bands_org.shape[2],))-1
        for i in range(pred_bands_org.shape[2]):
            b_max=np.max(pred_bands_org[:,:,i])
            b_min=np.min(pred_bands_org[:,:,i])
            min_arr[i] = b_min
            max_arr[i] = b_max
        # print("--- %s seconds at 3---" % (time.time() - start_time))
        # if np.sum(min_arr>0)==0 and np.sum(max_arr>0)==0:
        #     return

        for i in range(pred_bands_org.shape[2]):
            b_min = min_arr[i]
            b_max = max_arr[i]
            pred_bands_org[:,:,i]=(pred_bands_org[:,:,i]-b_min)/(b_max-b_min)
        # print("--- %s seconds ---at 4" % (time.time() - start_time))
        if brows<96:
            temp = pred_bands_org.copy()
            # ctemp = cloud_matrix.copy()
            nof = pred_bands_org.shape[2]

            pred_bands_org = np.zeros((96,bcols,nof))
            pad_ind = int((96-brows)/2)
            for i in range(nof):
                pred_bands_org[pad_ind:pad_ind+brows,:,i] = temp[:,:,i]

            # cloud_matrix = np.zeros((96,bcols,2))+2
            # pad_ind = int((96-brows)/2)
            # for i in range(2):
            #     cloud_matrix[pad_ind:pad_ind+brows,:,i] = ctemp[:,:,i]

            brows = 96

        if bcols<96:

            temp = pred_bands_org.copy()
            # ctemp = cloud_matrix.copy()
            nof = pred_bands_org.shape[2]

            pred_bands_org = np.zeros((brows,96,nof))
            pad_ind = int((96-bcols)/2)
            for i in range(nof):
                pred_bands_org[:,pad_ind:pad_ind+bcols,i] = temp[:,:,i]

            # cloud_matrix = np.zeros((brows,96,2))
            # pad_ind = int((96-bcols)/2)
            # for i in range(2):
            #     cloud_matrix[:,pad_ind:pad_ind+bcols,i] = ctemp[:,:,i]
            bcols = 96
        # print("--- %s seconds --- at 5" % (time.time() - start_time))
        # if os.path.isfile(img_dir + boxid + '/M' + fname[0:-4] + '.npy')==False or os.path.isfile(img_dir + boxid + '/C' + fname[0:-4] + '.npy')==False or os.path.isfile(img_dir + boxid + '/O' + fname[0:-4] + '.npy')==False:
        #     pred_labels_cor = np.zeros((brows,bcols))+3
        #     pred_labels_cloud = np.zeros((brows,bcols))+3
        #     pred_labels = np.zeros((brows,bcols))+3
        # else:
        #
        #     pred_labels_cloud = np.load(img_dir + boxid + '/M' + fname[0:-4] + '.npy')
        #     pred_labels = np.load(img_dir + boxid + '/C' + fname[0:-4] + '.npy')
        #     pred_labels_cor = np.load(img_dir + boxid + '/F' + fname[0:-4] + '.npy')


        if os.path.isfile(img_dir + boxid + '/A' + fname[0:-4] + '.npy')==False or os.path.isfile(img_dir + boxid + '/O' + fname[0:-4] + '.npy')==False:
            pred_labels_cor = np.zeros((brows,bcols))+3
            pred_labels_cloud = np.zeros((brows,bcols))+3
            pred_labels = np.zeros((brows,bcols))+3
        else:

            pred_labels_cloud = np.load(img_dir + boxid + '/A' + fname[0:-4] + '.npy')
            pred_labels = pred_labels_cloud.copy()
            # pred_labels = np.load(img_dir + boxid + '/C' + fname[0:-4] + '.npy')
            pred_labels_cor = np.load(img_dir + boxid + '/O' + fname[0:-4] + '.npy')

        # T = np.zeros((1,98))
        area_info = np.load(img_dir + boxid + '/area_info.npy')
        area_ts = area_info[0]
        dates_ts = area_info[1]
        # print("--- %s seconds ---at 6" % (time.time() - start_time))
        plot_pred_cloud(pred_bands_org.copy(),np.expand_dims(pred_labels,axis=2),np.expand_dims(pred_labels_cloud,axis=2),np.expand_dims(pred_labels_cor,axis=2),img_dir + boxid + '/' + fname[0:-4] + '.png')
        plot_pred_video(pred_bands_org.copy(),np.expand_dims(pred_labels_cor,axis=2),area_ts,dates_ts,img_dir + boxid + '/' + fname[0:-4] + '.png')
        # print("--- %s seconds --- at 7" % (time.time() - start_time))
print(data_dir)
flist = glob.glob(img_dir + prefix + '/S2*_A00000.tif')
print('Number of files: ' + str(len(flist)))
print('preparing area time series...')
area_info = CreateAreaTimeseries(glob.glob(img_dir + prefix + '/S2*_A00000.tif'),img_dir)
# print(len(area_info[1]),len(np.unique(area_info[1])))
# plt.figure()
# plt.plot(area_info[0])
# plt.show()
np.save(img_dir + prefix + '/area_info',area_info)
# sys.exit()

tasks = []
# plot_pred_video([],[],[],[])
# sys.exit()
for fpath in flist:
    fname = fpath[fpath.rfind('/')+1:]
    boxid = fpath.split('/')[-2]
    # tasks.append((fpath,img_dir,1))
    # if os.path.isdir(img_dir + boxid)==False:
        # os.mkdir(img_dir + boxid)
    # print(img_dir)
    pred_fun2(fpath,img_dir,1)
# pool = mp.Pool(processes=10,maxtasksperchild=10)
# results = [pool.apply_async(pred_fun2, t ) for t in tasks]
# for result in results:
#     result.get()
