import os
import sys
import gdal
import glob
import numpy as np

def ReadSentinel2Image(ipath):
    ds = gdal.Open(ipath)
    delv = np.array(ds.GetRasterBand(1).ReadAsArray())

    brows = delv.shape[0]
    bcols = delv.shape[1]
    b_arr = np.arange(1,18)
    nof = len(b_arr)
    pred_bands_org=np.zeros((brows,bcols,nof),float)
    # b_arr = np.array([8,4,3])
    for b in range(pred_bands_org.shape[2]):
        pred_bands_org[:,:,b]= np.array(ds.GetRasterBand(int(b_arr[b])).ReadAsArray())

    ds = None
    return pred_bands_org

data_dir = sys.argv[1]
img_dir = sys.argv[2]
cinfo = sys.argv[3]
rasterFormat = 'GTiff' # for now assuming output format is going to GTiff
rasterDriver = gdal.GetDriverByName(rasterFormat)

if '/' in cinfo:
    prefix = cinfo[0:-4]
    prefix = prefix[prefix.rfind('/')+1:]
else:
    prefix = cinfo
boxid = prefix

data_dir = data_dir + prefix + '/'
img_dir = img_dir + prefix + '/'

# selecting all classification files
imgs = glob.glob(img_dir + 'M*' + '*.npy')
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

    imgs = glob.glob(img_dir + 'M*' + fname + '*.npy')
    # tname = imgs[0].split('/')[-1]


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
    #initialize merged data
    fmap = np.load(imgs[sinds[0]]).copy()
    fmap[:,:] = 3
    spath = imgs[sinds[0]]
    sname = spath.split('/')[-1]
    sname = sname[1:-4]
    spath = data_dir + sname + '.tif'
    fimg = ReadSentinel2Image(spath)
    ds = gdal.Open(spath)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None




    try:

        for j in range(0,len(imgs)):
            img = imgs[sinds[j]]
            mask_arr = np.load(img)

            spath = imgs[sinds[j]]
            sname = spath.split('/')[-1]
            sname = sname[1:-4]
            spath = data_dir + sname + '.tif'
            data_arr = ReadSentinel2Image(spath)

            new_inds = np.logical_and(fmap==3,mask_arr<=1)
            fmap[new_inds] = mask_arr[new_inds]
            for b in range(fimg.shape[2]):
                temp = fimg[:,:,b].copy()
                curr = data_arr[:,:,b]
                temp[new_inds] = curr[new_inds]
                fimg[:,:,b] = temp





    except:
        print('size mismatch')
        os.system('rm -rf ' + img_dir + '/A*.npy')
        os.system('rm -rf ' + img_dir + '/S2*_A00000.tif')

        # sys.exit()
        break
    sname = sname[0:-6] + 'A00000' + '.tif'
    mds = rasterDriver.Create(img_dir + sname,cols,rows,int(fimg.shape[2]),gdal.GDT_UInt16)
    mds.SetGeoTransform(geotransform)
    mds.SetProjection(projection)
    for b in range(fimg.shape[2]):
        mds.GetRasterBand(b+1).WriteArray(fimg[:,:,b])
    mds = None
    np.save(img_dir + '/A' + sname[0:-4],fmap)
    # np.save(img_dir + '/F' + tname[1:-4],fmap)
