import pandas as pd
import numpy as np
import random
import multiprocessing as mp
import gdal
from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.initializers
import models
import dproc
import matplotlib.pyplot as plt
import sys
import os

run_task = int(sys.argv[1])
iter_num = int(sys.argv[2])
iter_num2 = int(sys.argv[3])
exp_dir = os.getcwd() + '/'
info_dir = '/home/kumarv/khand035/Projects/MINT/river-depth-estimation/scripts/'
if os.path.isdir(exp_dir)==False:
    os.mkdir(exp_dir)


if run_task==1:
    # Sentinel-2 img to img training module
    auto_model = exp_dir + 's2img_s2img_exp4_run_num_' + str(iter_num) + '.hdf5'


    jf = pd.read_csv(info_dir + 'dataset_08232019.txt',names=['name','cscore','iscore','rows','cols'])
    cscore = jf['cscore'].values
    iscore = jf['iscore'].values
    rows = jf['rows'].values
    cols = jf['cols'].values
    cth = 5
    cond = np.logical_and(cscore+iscore<cth,rows>=96)
    cond = np.logical_and(cond,cols>=96)
    print(jf.shape[0],np.sum(cscore<cth),np.sum(cond))


    tasks = []
    for j in range(jf.shape[0]):
        if cond[j]==False:
            continue
        tasks.append(jf['name'][j])

    random.shuffle(tasks)
    N = len(tasks)
    batch_size = 32
    steps_per_epoch = N/batch_size
    train_imgs = tasks[0:N]
    np.save(auto_model[0:-5],train_imgs)

    model_img = models.img_img()
    model_checkpoint = ModelCheckpoint(auto_model, monitor='loss',verbose=1, save_best_only=True)

    myGene = dproc.batch_generator(train_imgs,batch_size,rtype='image-image')
    history=model_img.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=10,callbacks=[model_checkpoint],use_multiprocessing=True,workers=5)

if run_task==2:

    auto_model = exp_dir + 's2img_s2img_exp4_run_num_1.hdf5'
    seg_model = exp_dir + 'random_s2img_s2lab_exp8.hdf'



    # training of segmentation model
    jf = pd.read_csv(info_dir + 'labels_08232019.txt',index=False)
    cscore = jf['cscore'].values
    iscore = jf['iscore'].values
    rows = jf['rows'].values
    cols = jf['cols'].values
    cth = 101
    cond = np.logical_and(cscore+iscore<cth,rows>=96)
    cond = np.logical_and(cond,cols>=96)

    tasks = []
    for j in range(jf.shape[0]):
        if cond[j]==False:
            continue
        tasks.append(jf['name'][j])

    print(len(tasks))
    random.shuffle(tasks)
    # ReadImageStack(tasks[0:10],isauto=0)
    N = len(tasks)
    batch_size = 32
    steps_per_epoch = N/batch_size
    train_imgs = tasks[0:N]

    # model = models.img_label(pretrained_weights=auto_model,type='frozen')
    # model_checkpoint = ModelCheckpoint(seg_model_frozen, monitor='loss',verbose=1, save_best_only=True)
    #
    # myGene = dproc.batch_generator(train_imgs,batch_size,rtype='image-label')
    # history=model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=10,callbacks=[model_checkpoint],use_multiprocessing=True,workers=5)
    #
    # model = models.img_label(pretrained_weights=auto_model,type='init')
    # model_checkpoint = ModelCheckpoint(seg_model_init, monitor='loss',verbose=1, save_best_only=True)
    #
    # myGene = dproc.batch_generator(train_imgs,batch_size,rtype='image-label')
    # history=model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=10,callbacks=[model_checkpoint],use_multiprocessing=True,workers=5)

    model = models.img_label(pretrained_weights=None,type='random')
    model_checkpoint = ModelCheckpoint(seg_model, monitor='loss',verbose=1, save_best_only=True)

    myGene = dproc.batch_generator(train_imgs,batch_size,rtype='image-label')
    history=model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=10,callbacks=[model_checkpoint],use_multiprocessing=True,workers=5)


    # os.system('cp models_exp1.py ' + exp_dir)
    # os.system('cp dproc.py ' + exp_dir)
    # os.system('cp Experiment1.py ' + exp_dir)


if run_task==3:
    #------------- Sentinel-2 prediction module

    # auto_model = exp_dir + 's2img_s2img_exp2.hdf5'
    seg_model = exp_dir + 'random_s2img_s2lab_exp5_bayesian.hdf'
    imgdir = exp_dir + 'pred_figures_bayesian/'
    Nlimit = 100


    if os.path.isdir(imgdir)==False:
        os.mkdir(imgdir)

    jf = pd.read_csv(info_dir + 'dataset_08232019.txt',names=['name','cscore','iscore','rows','cols'])
    # print(jf)
    cscore = jf['cscore'].values
    iscore = jf['iscore'].values
    rows = jf['rows'].values
    cols = jf['cols'].values
    cth = 5
    cond = np.logical_and(cscore+iscore<cth,rows>=96)
    cond = np.logical_and(cond,cols>=96)
    print(jf.shape[0],np.sum(cscore<cth),np.sum(cond))


    tasks = []

    for j in range(jf.shape[0]):
        if cond[j]==False:
            continue
        tasks.append(jf['name'][j])

    print(len(tasks))
    random.shuffle(tasks)
    N = min([len(tasks),Nlimit])
    batch_size = 64
    steps_per_epoch = N/batch_size
    train_imgs = tasks[0:N]


    model = models.img_label(pretrained_weights=seg_model,type='init')
    print('display begins')
    for counter in range(0,N,batch_size):
        print(counter)
        ilist = train_imgs[counter:min([N,counter + batch_size])]
        X = dproc.ReadImageStack(ilist,rtype='image')
        for bays in range(10):
            Y = model.predict(X)
        # print(Y.shape)
        tasks = []
        for j in range(len(ilist)):
            # print(counter,j)
            imgpath = ilist[j]
            # print(model.evaluate(np.expand_dims(X[j,:,:,:],axis=0),np.expand_dims(X[j,:,:,:],axis=0),verbose=0))
            # Z = np.round(model.evaluate(np.expand_dims(X[j,:,:,:],axis=0),np.expand_dims(X[j,:,:,:],axis=0),verbose=0)[0]*10000.0)
            boxname = imgpath.split('/')[-2]
            # imgname = str(int(Z)) + '_' + boxname + '_' + imgpath[imgpath.rfind('/')+1:-4] + '.png'
            imgname = boxname + '_' + imgpath[imgpath.rfind('/')+1:-4] + '.png'
            tasks.append((X[j,:,:,:],Y[j,:,:,:],imgdir + imgname))

        pool = mp.Pool(processes=10,maxtasksperchild=10)
        results = [pool.apply_async(dproc.plot_pred, t ) for t in tasks]
        for result in results:
            result.get()
