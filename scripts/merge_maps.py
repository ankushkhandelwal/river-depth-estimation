import os
import sys
import gdal
import glob
import numpy as np
data_dir = sys.argv[1]
cinfo = sys.argv[2]

prefix = cinfo[0:-4]
if '/' in prefix:
    prefix = prefix[prefix.rfind('/')+1:]

data_dir = data_dir + prefix + '/'

# selecting all classification files
imgs = glob.glob(data_dir + 'O*' + '*.npy')
# print(len(imgs))

fnames = []
print('Merging classification maps...')
# extracting date strings
for img in imgs:
    tname = img[img.rfind('/')+1:]
    # print tname, tname[-34:-26]
    fnames.append(tname[-34:-26])

# sorting dates
fnames = list(set(fnames))
fnames.sort()
# print fnames
# sys.exit()
# merging classification maps for each date
for i in range(0,len(fnames)):
    fname = fnames[i]
    print(fname)
    # if os.path.isfile(data_dir + '/F3_' + fname + '.npy')==True:
    #     # print('already exists')
    #     continue

    imgs = glob.glob(data_dir + 'O*' + fname + '*.npy')
    #print fname
    # print fname, len(imgs)


    # storing sensor type and valid pixel count for each map
    stype = np.zeros((len(imgs),))
    gcnt = np.zeros((len(imgs),))
    for j in range(0,len(imgs)):
        img = imgs[j]
        # tname = img[img.rfind('/')+1:]
        stype[j] = 2

        mask_arr = np.load(img)
        gcnt[j] = np.sum(mask_arr<=1)
        rows = mask_arr.shape[0]
        cols = mask_arr.shape[1]


    # giving first priority to Sentinel-2 maps
    # then number of valid pixels
    sinds = np.lexsort((gcnt,stype))

    # f,(ax1,ax2,ax3) = plt.subplots(1,3)

    # merging classification maps in the order
    fmap = np.load(imgs[sinds[0]]).copy()
    try:

        for j in range(0,len(imgs)):
            img = imgs[sinds[j]]
            # print img
            mask_arr = np.load(img)
            new_inds = np.logical_and(fmap==3,mask_arr<=1)
            fmap[new_inds] = mask_arr[new_inds]

    except:
        print('size mismatch')
        os.system('rm -rf ' + data_dir + '/F3_*.npy')
        # sys.exit()
        break
    np.save(data_dir + '/F3_' + fname,fmap)
