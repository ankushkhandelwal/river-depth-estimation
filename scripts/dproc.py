# merge cloud and size info into a single panda frame
import pandas as pd
import numpy as np
import random
# import multiprocessing as mp
import gdal
# from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
# import model_library


def ReadNDWIImage(imgpath,rows,cols,idn):
    ds=gdal.Open(imgpath)


def ReadSentinel1Image(imgpath,rows,cols,idn):
    ds=gdal.Open(imgpath)
    delv = np.array(ds.GetRasterBand(2).ReadAsArray())
    # plt.figure()
    # plt.imshow(delv)
    # plt.show()
    # plt.close()
    brows = delv.shape[0]
    bcols = delv.shape[1]
    pad_row = int((brows-rows)/2)
    pad_col = int((bcols-cols)/2)
    delv = delv.astype(float)
    temp = delv.copy()
    temp = temp.astype(float)
    temp[temp==0] = np.nan
    mean = np.nanmean(temp)
    std = np.nanstd(temp)

    delv = (delv - mean)*1.0/std
    delv = delv - np.min(delv[np.isnan(temp)==0])
    delv[np.isnan(temp)] = np.max(delv[np.isnan(temp)==0])
    # print(mean,std,np.max(delv))
    delv = delv*1.0/np.max(delv[np.isnan(temp)==0])

    # print(mean,std,np.sum(np.isnan(delv)))
    vh = delv.copy()

    # vh = delv*1.0/255
    # vh[vh==0]=1
    matrix_rp_center=vh[pad_row:pad_row+rows,pad_col:pad_col+cols]
    temp = np.expand_dims(matrix_rp_center,axis=2)
    # print('in reading')
    # plt.figure()
    # plt.imshow(np.squeeze(temp))
    # plt.show()
    # plt.close()
    return temp,idn


def ReadImage(imgpath,rows,cols,barr,idn):
    # print(idn)
    ds = gdal.Open(imgpath)
    delv = np.array(ds.GetRasterBand(1).ReadAsArray())
    brows = delv.shape[0]
    bcols = delv.shape[1]
    matrix_rp=np.zeros((brows,bcols,barr.shape[0]),float)
    for b in range(matrix_rp.shape[2]):
        matrix_rp[:,:,b]= np.array(ds.GetRasterBand(int(barr[b]+1)).ReadAsArray())

    pad_row = int((brows-rows)/2)
    pad_col = int((bcols-cols)/2)
    matrix_rp_center=matrix_rp[pad_row:pad_row+rows,pad_col:pad_col+cols,:]
    matrix_rp_norm=matrix_rp_center.copy()
    for b in range(matrix_rp_norm.shape[2]):
        b_max=np.max(matrix_rp_center[:,:,b])
        b_min=np.min(matrix_rp_center[:,:,b])
        # print(b,b_min,b_max)
        matrix_rp_norm[:,:,b]=(matrix_rp_center[:,:,b]-b_min)/(b_max-b_min)

    return matrix_rp_norm,idn


def ReadSentinel1Labels(imgpath,rows,cols,idn):
    # print(idn)
    imgpath = imgpath[0:-4] + '.npy'
    delv = np.load(imgpath)
    brows = delv.shape[0]
    bcols = delv.shape[1]
    pad_row = int((brows-rows)/2)
    pad_col = int((bcols-cols)/2)
    labels=delv[pad_row:pad_row+rows,pad_col:pad_col+cols]
    labels = np.expand_dims(labels,axis=2)
    return labels,idn


def ReadLabels(imgpath,rows,cols,idn):
    # print(idn)
    imgpath = imgpath[0:-4] + '.npy'
    delv = np.load(imgpath)
    brows = delv.shape[0]
    bcols = delv.shape[1]
    pad_row = int((brows-rows)/2)
    pad_col = int((bcols-cols)/2)
    labels=delv[pad_row:pad_row+rows,pad_col:pad_col+cols]
    labels = np.expand_dims(labels,axis=2)
    return labels,idn

def ReadImageStack(ilist,rtype='image-image'):

    rows = 96
    cols = 96
    barr = np.array([1,2,3,4,5,6,7,11,12]).astype(int)
    N = len(ilist)
    nof = barr.shape[0]

    if rtype=='image-label':
        Feature = np.zeros((N,rows,cols,nof))
        Labels = np.zeros((N,rows,cols,1))
        for i in range(N):
            Feature[i,:,:,:] = ReadImage(ilist[i],rows,cols,barr,i)[0]
            Labels[i,:,:,:] = ReadLabels(ilist[i],rows,cols,i)[0]

        return Feature,Labels

    if rtype=='image-image':
        Feature = np.zeros((N,rows,cols,nof))

        for i in range(N):
            Feature[i,:,:,:] = ReadImage(ilist[i],rows,cols,barr,i)[0]

        return Feature,Feature


    if rtype=='image':
        Feature = np.zeros((N,rows,cols,nof))

        for i in range(N):
            Feature[i,:,:,:] = ReadImage(ilist[i],rows,cols,barr,i)[0]

        return Feature

    if rtype=='gray-label':
        barr = np.array([2]).astype(int)
        nof = barr.shape[0]
        Feature = np.zeros((N,rows,cols,1))
        Labels = np.zeros((N,rows,cols,1))
        for i in range(N):
            Feature[i,:,:,:] = ReadSentinel1Image(ilist[i],rows,cols,i)[0]
            Labels[i,:,:,:] = ReadSentinel1Labels(ilist[i],rows,cols,i)[0]

        return Feature,Labels


    if rtype=='gray':
        barr = np.array([2]).astype(int)
        nof = barr.shape[0]
        Feature = np.zeros((N,rows,cols,1))
        Labels = np.zeros((N,rows,cols,1))
        for i in range(N):
            Feature[i,:,:,:] = ReadSentinel1Image(ilist[i],rows,cols,i)[0]
            # print(Feature[i,:,:,:].shape)
            # plt.figure()
            # plt.imshow(np.squeeze(Feature[i,:,:,:]))
            # plt.show()
            # plt.close()
            # Labels[i,:,:,:] = ReadSentinel1Labels(ilist[i],rows,cols,i)[0]

        return Feature

    if rtype=='gray-gray':
        barr = np.array([2]).astype(int)
        nof = barr.shape[0]
        Feature = np.zeros((N,rows,cols,1))
        Labels = np.zeros((N,rows,cols,1))
        for i in range(N):
            Feature[i,:,:,:] = ReadSentinel1Image(ilist[i],rows,cols,i)[0]
            # print(Feature[i,:,:,:].shape)
            # plt.figure()
            # plt.imshow(np.squeeze(Feature[i,:,:,:]))
            # plt.show()
            # plt.close()
            # Labels[i,:,:,:] = ReadSentinel1Labels(ilist[i],rows,cols,i)[0]

        return Feature,Feature


    if rtype=='ndwi-label':
        Feature = np.zeros((N,rows,cols,nof))
        Labels = np.zeros((N,rows,cols,1))
        for i in range(N):
            Feature[i,:,:,:] = ReadNDWIImage(ilist[i],rows,cols,barr,i)[0]
            Labels[i,:,:,:] = ReadLabels(ilist[i],rows,cols,i)[0]

        return Feature,Labels


def batch_generator(ilist,batch_size,rtype):

    samples_per_epoch = len(ilist)
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    # print('inside')
    while(1):
        # print(batch_size*counter)
        # print(batch_size*counter,len(ilist[batch_size*counter:batch_size*(counter+1)]))
        X_batch,Y_batch = ReadImageStack(ilist[batch_size*counter:batch_size*(counter+1)],rtype)
        # print(X_batch.shape)
        counter += 1
        yield X_batch,Y_batch

        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            random.shuffle(ilist)
            counter = 0


def plot_pred(X,Y,savepath):
    print(X.shape,Y.shape)
    rgb_map=np.zeros((X.shape[0],X.shape[1],3),float)
    rgb_map[:,:,0]= X[:,:,8]
    rgb_map[:,:,1]= X[:,:,6]
    rgb_map[:,:,2]= X[:,:,2]
    print(rgb_map.shape)
    plt.figure()
    plt.show()
    # f,(ax1,ax2) = plt.subplots(1,2)
    # ax1.imshow(rgb_map)
    # ax2.imshow(np.squeeze(Y[:,:,:]))
    # # if savepath=='':
    # #     print('not saving')
    # #     plt.show()
    # # else:
    # #     f.savefig(savepath)
    plt.close()


# def plot_pred_Variance(X,Y,Z,savepath):
#     # print(X.shape,Y.shape)
#     rgb_map=np.zeros((X.shape[0],X.shape[1],3),float)
#     rgb_map[:,:,0]= X[:,:,8]
#     rgb_map[:,:,1]= X[:,:,6]
#     rgb_map[:,:,2]= X[:,:,2]
#     f,(ax1,ax2,ax3) = plt.subplots(1,3)
#     ax1.imshow(rgb_map)
#     tY = np.squeeze(Y[:,:,:])
#     tY[0,0] = 0
#     tY[0,1] = 1
#
#     tZ = np.squeeze(Z[:,:,:])
#     tZ[0,0] = 0
#     tZ[0,1] = 0.3
#
#
#     # tW = np.squeeze(W[:,:,:])
#     # tW[0,0] = 0
#     # tW[0,1] = 1
#
#
#     ax2.imshow(tY)
#     ax3.imshow(tZ)
#     # ax4.imshow(tW)
#
#     if savepath=='':
#         print('not saving')
#         plt.show()
#     else:
#         f.savefig(savepath)
#     plt.close()



def plot_recon(X,Y,savepath):
    # print(X.shape,Y.shape)
    rgb_map=np.zeros((X.shape[0],X.shape[1],3),float)
    rgb_map[:,:,0]= X[:,:,8]
    rgb_map[:,:,1]= X[:,:,6]
    rgb_map[:,:,2]= X[:,:,2]

    rgb_map2=np.zeros((Y.shape[0],Y.shape[1],3),float)
    rgb_map2[:,:,0]= Y[:,:,8]
    rgb_map2[:,:,1]= Y[:,:,6]
    rgb_map2[:,:,2]= Y[:,:,2]

    f,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(rgb_map)
    ax2.imshow(rgb_map2)
    if savepath=='':
        print('not saving')
        plt.show()
    else:
        f.savefig(savepath)
    plt.close()



def plot_sent1_pred(X,Y,savepath):
    # print(X.shape,Y.shape)
    f,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(np.squeeze(X[:,:,:]))
    ax2.imshow(np.squeeze(Y[:,:,:]))
    f.savefig(savepath)
    plt.close()


def ReadTxtFile(filename):
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


# This code takes an image and breaks in into small pieces of given size
def SplitImage(imgpath,out_dir,rows,cols,idn):

    rasterFormat = 'GTiff' # for now assuming output format is going to GTiff
    rasterDriver = gdal.GetDriverByName(rasterFormat)
    ds = gdal.Open(imgpath)
    delv = np.array(ds.GetRasterBand(1).ReadAsArray())
    nof  = ds.GetRasterCount()
    brows = delv.shape[0]
    bcols = delv.shape[1]
    pred_bands_org=np.zeros((brows,bcols,nof),float)
    for b in range(matrix_rp.shape[2]):
        pred_bands_org[:,:,b]= np.array(ds.GetRasterBand(b+1).ReadAsArray())


    if brows<rows:
        temp = pred_bands_org.copy()
        nof = pred_bands_org.shape[2]

        pred_bands_org = np.zeros((rows,bcols,nof))
        pad_ind = int((rows-brows)/2)
        for i in range(nof):
            pred_bands_org[pad_ind:pad_ind+brows,:,i] = temp[:,:,i]

        brows = rows

    if bcols<cols:

        temp = pred_bands_org.copy()
        nof = pred_bands_org.shape[2]

        pred_bands_org = np.zeros((brows,cols,nof))
        pad_ind = int((cols-bcols)/2)
        for i in range(nof):
            pred_bands_org[:,pad_ind:pad_ind+bcols,i] = temp[:,:,i]

        bcols = cols


    rarr = np.arange(0,brows,24)
    rarr = rarr[0:-5]
    rarr = np.append(rarr,brows-rows)
    carr = np.arange(0,bcols,24)
    carr = carr[0:-5]
    carr = np.append(carr,bcols-cols)
    ctr = 0
    for r in rarr:
        for c in carr:
            # print(r,c,ctr)
            # print(ctr)
            pred_bands=pred_bands_org[r:r+rows,c:c+cols,:].copy()
            imgname = imgpath.split('/')[-1]

            outputFile = out_dir + imgname[0:-4] + '_patch_' + str(ctr) + '.tif'
            mds = rasterDriver.Create(outputFile,cols,rows,nof,gdal.GDT_UInt16)
            for j in range(nof):
                mds.GetRasterBand(j+1).WriteArray(pred_bands[:,:,j])
            mds = None
            ctr = ctr + 1

def cloud_pred(full_band_matrix):

    full_band_matrix=full_band_matrix*1.0/10000
    band_matrix_input=full_band_matrix[:,:,[0,1,3,4,7,8,9,10,11,12]]
    band_matrix_input = np.reshape(band_matrix_input,(1,)+band_matrix_input.shape)
    cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
    cloud_matrix=np.zeros((band_matrix_input.shape[1],band_matrix_input.shape[2],2),float)
    cloud_matrix[:,:,0] = cloud_detector.get_cloud_probability_maps(np.array(band_matrix_input))
    cloud_matrix[:,:,1] = cloud_detector.get_cloud_masks(np.array(band_matrix_input))
    # plt.figure()
    # plt.imshow(cloud_matrix[:,:,1])
    return cloud_matrix
    # np.save(output_filename,cloud_matrix)
