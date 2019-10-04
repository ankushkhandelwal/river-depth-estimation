import os
import sys
import numpy as np
import skimage
import skimage.measure
import skimage.morphology
import glob
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from scipy import ndimage
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import gdal,ogr
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import pandas as pd


def ExtractPiecewithCenterLine(labp,cp):
    # print(labp.shape,cp)
    for i in range(0,labp.shape[0]):
        if labp[i]==0:
            continue
        labp[0:i] = 1
        j = np.where(labp==0)[0][0]
        if i<=cp and j>cp:
            labp[0:i] = 0
            labp[i:j] = 1
            labp[j:] = 0
            return labp
        else:
            labp[i:j] = 0

    return labp



def CalculateLabelProfilePurity(labp):

    i = np.where(labp==1)[0][0]
    j = np.where(labp==1)[0][-1]
    return np.sum(labp)*1.0/(j-i+1)


def CreateTimeSeries(data_dir,boxn,sec):
    cur_csv = data_dir + boxn  + '-sec-' + sec + '.csv'
    if os.path.isfile(cur_csv)==False:
        return -1,-1,-1,-1,-1,-1
    df = pd.read_csv(cur_csv,names=['date','width','dl','dr','da','ec','ledge','medge','redge','cline','lat','lon'],delimiter=',')
    widths = df['width'].values
    dates = df['date'].values
    dl = df['dl'].values
    dr = df['dr'].values
    da = df['da'].values

    xticklabels = []
    for i in range(df.shape[0]):
        xticklabels.append(df.iloc[i]['date'][5:10])

    widths = widths.astype(float)
    dl = dl.astype(float)
    dr = dr.astype(float)
    da = da.astype(float)

    bad_inds = widths==-1
    # bad_inds = np.logical_and(bad_inds,dr==0)
    widths[bad_inds] = np.nan
    dl[bad_inds] = np.nan
    dr[bad_inds] = np.nan
    da[bad_inds] = np.nan
    return da,dl,dr,widths,xticklabels,1

def CreateSinglePixelGeoTiff(i,j,geobase,temp_tif):
    os.system('cp ' + geobase + ' ' + temp_tif)
    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(temp_tif,1)
    band = dataset.GetRasterBand(1).ReadAsArray()
    band[:,:,] = 0
    band[i,j] = 1
    dataset.GetRasterBand(1).WriteArray(band)
    dataset=None
    os.system('gdalwarp -q -overwrite ' + temp_tif + ' ' + temp_tif[0:-4] + '_wgs84.tif -t_srs EPSG:4326')
    dataset = gdal.Open(temp_tif[0:-4] + '_wgs84.tif',1)
    band = dataset.GetRasterBand(1).ReadAsArray()
    ctr = 0
    for i in range(band.shape[0]):
        for j in range(band.shape[1]):
            if band[i,j]==1:
                si = i
                sj = j
                ctr = ctr + 1
    # print('Number of Matches: ' + str(ctr))
    if ctr==0:
        si = -1
        sj = -1
    return si,sj,temp_tif[0:-4] + '_wgs84.tif'

def RecoverLabels(pmasks,masks,emask):
    rows,cols,numt = masks.shape

    for i in range(0,numt):
        cur_mask = masks[:,:,i].copy()
        cur_pmask = pmasks[:,:,i].copy()
        cur_pmask[emask] = cur_mask[emask]
        pmasks[:,:,i] = cur_pmask

    return pmasks

def SelectLargestComponent(masks,jrc_mask):
    rows,cols,numt = masks.shape
    new_masks = np.zeros((rows,cols,numt))

    for i in range(0,numt):
        cur_mask = masks[:,:,i]==1
        cur_labels = skimage.measure.label(cur_mask,background=0)
        # f,(ax1,ax2,ax3) = plt.subplots(1,3)
        # ax1.imshow(cur_mask)
        # ax2.imshow(cur_labels)
        cur_labels_count = cur_labels.copy()
        cur_labels_count[jrc_mask==0] = 0
        # ax3.imshow(cur_labels_count)
        # plt.show()

        if np.max(cur_labels)==0 or np.max(cur_labels_count)==0:
            lcc=-2
        else:
            lcc = np.argmax(np.bincount(cur_labels_count.flatten())[1:])

        new_mask = cur_labels==lcc+1
        new_mask = new_mask.astype('int')
        new_mask[masks[:,:,i]>1] = 2
        new_masks[:,:,i] = new_mask

    return new_masks

def pix2coord(x,y,image):
    ds = gdal.Open(image,0)
    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up.
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()

    """Returns global coordinates from pixel x, y coords"""
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def GetCrossSectionProfiles4(csec,cline,lmask,delv):
    row,cols = csec.shape
    # print(np.sum(csec[cline>0]))
    # csec[cline>0] = csec[cline>0]+2
    # plt.figure()
    # plt.imshow(csec)
    # plt.show()
    if delv.shape[0]!=rows or delv.shape[1]!=cols:
        return -1,-1,-1,-1
    linds = np.where(csec==1)
    rinds = linds[0]
    cinds = linds[1]
    sinds = np.argsort(np.minimum(cinds,rinds))
    curr = rinds[sinds[0]]
    curc = cinds[sinds[0]]
    sinds = sinds.astype(int)
    sinds[1:] = -1
    bad_inds = np.zeros((rinds.shape[0],)).astype(int)
    bad_inds[sinds[0]] = 1
    #print(sinds[0],curr,curc,bad_inds[sinds[0]])
    for i in range(1,rinds.shape[0]):

        cur_diff = np.abs(rinds-curr) + np.abs(cinds-curc)
        cur_diff[bad_inds==1] = np.max(cur_diff)+1

        sinds[i] = int(np.argmin(cur_diff))
        bad_inds[sinds[i]] = 1
        #print(bad_inds)
        curr = rinds[sinds[i]]
        curc = cinds[sinds[i]]


    labp = np.zeros((rinds.shape[0],))
    elep = np.zeros((rinds.shape[0],))
    rmask = np.zeros((row,cols))
    ctr = 0
    # print(rinds.shape)
    cp = -1
    for p in range(0,rinds.shape[0]):
        labp[p] = lmask[rinds[sinds[p]],cinds[sinds[p]]]
        elep[p] = delv[rinds[sinds[p]],cinds[sinds[p]]]
        rmask[rinds[sinds[p]],cinds[sinds[p]]] = p+1
        if cline[rinds[sinds[p]],cinds[sinds[p]]]==1:
            ctr = ctr + 1
            cp = p
    if ctr>1:
        cp = -2
    return labp,elep,cp,rmask



def SelectMaximumExtent(frac,pmasks):
    score = np.zeros((pmasks.shape[2],))
    for i in range(0,pmasks.shape[2]):
        score[i] = np.sum(frac!=pmasks[:,:,i])
    return np.argmin(score)

def PrepareCrossSections3(image,geobase):
    sr = 5 # start row
    jump = 3
    sw = 5 # section winow width
    nc = 15
    row1, col1 = image.shape
    linds = np.where(image==1)
    rinds = linds[0]
    cinds = linds[1]
    sinds = np.lexsort((rinds,cinds))
    rinds = rinds[sinds]
    cinds = cinds[sinds]
    if rinds.shape[0]==0:
        cross_image = np.zeros((row1,col1,nc)) # store each cross section separately
        coord_info = np.zeros((nc,2)) # store each cross section separately
        return cross_image, coord_info

    if (np.max(rinds) - np.min(rinds)) > np.max(cinds) - np.min(cinds):
        pdist = rinds
        drange = row1*1.0/(nc+1)
    else:
        pdist = cinds
        drange = col1*1.0/(nc+1)

    cross_image = np.zeros((row1,col1,nc)) # store each cross section separately
    coord_info = np.zeros((nc,2)) # store each cross section separately
    ctr = 0
    plist = []
    for i in range(0,nc):
        cur_image = np.zeros((row1,col1))
        cur_dist = drange + i*drange
        cur_diff = np.abs(pdist - cur_dist)
        p = np.argmin(cur_diff)
        if p in plist:
            continue
        plist.append(p)
        #p = int(drange+i*drange)
        i = rinds[p]
        j = cinds[p]
        i,j,geobase_wgs84 = CreateSinglePixelGeoTiff(i,j,geobase,boxNum + '.tif')

        if i<0:
            clat = 500
            clon = 500
        else:
            clat,clon = pix2coord(i,j,geobase_wgs84)

        cur_image[i-sw:i+sw,j-sw:j+sw] = image[i-sw:i+sw,j-sw:j+sw]
        # i,j is the pixel through which cross section should pass

        h, theta, d = hough_line(cur_image,theta=np.arange(-np.pi/2,np.pi/2,np.pi/360))

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d,min_distance=1,min_angle=1,threshold=0.99*np.max(h))):

            if angle>np.pi/2:
                angle = angle - np.pi/2
            else:
                angle = angle + np.pi/2

            dist = j*np.cos(angle)*1.0 + i*np.sin(angle)*1.0


            lin_image = np.zeros((row1,col1))
            # burning the line
            klist = np.arange(0,col1,0.1)
            for k in klist:
                y1 = (dist - k * np.cos(angle)) / np.sin(angle)
                y1_int = int(np.round(y1))
                k_int = int(np.round(k))
                if y1_int>=0 and y1_int<row1 and k_int>=0 and k_int<col1:
                    lin_image[y1_int,k_int] = 1

            # use the cross section if it passes the current center line pixel
            # exit after finding the first such cross section
            if lin_image[i,j]>0:
                lin_image = skimage.morphology.skeletonize(lin_image)
                cross_image[:,:,ctr] = lin_image
                coord_info[ctr,0] = clat
                coord_info[ctr,1] = clon
                ctr = ctr + 1
                break

    return cross_image,coord_info


def FitLine(x,y,tx):
    m = (y[1]-y[0])*1.0/(x[1]-x[0])
    c = y[0] - m*x[0]
    ty = np.zeros((len(tx),))
    for i in range(0,len(tx)):
        ty[i] = m*tx[i] + c
    return ty


def IterativeConvolution(elvp,sw):
    temp = np.where(elvp>0)[0]
    # print(temp)
    elvp[elvp==0] = elvp[temp[0]]
    elvp_arr = []
    while(sw>0):
        temp = np.convolve(elvp, np.ones((sw,))/sw, mode='same')
        temp[0:int(sw/2)] = -1
        temp[int(-sw/2):] = -1
        elvp_arr.append(temp)
        sw = int(sw/2)
    elvp_arr.append(elvp)
    felvp = elvp.copy()
    felvp[:] = -1
    ctr = 0
    while np.sum(felvp==-1)>0:
        cur_elvp = elvp_arr[ctr]
        need_inds = felvp==-1
        felvp[need_inds] = cur_elvp[need_inds]
        ctr = ctr+1
    return felvp


# os.system('rm -f ' + data_dir + '*.csv')
# os.system('rm -f ' + data_dir + '*.png')
# boxNum =
# base_dir = data_dir[0:u+1]
# if os.path.isfile(data_dir + data_name + '-vizlocal.csv')==True:
#     sys.exit()

# u = data_name.find('-')
# catNum = data_name[5:u]
#print(catNum)

#-------------main script--------------------
# base_dir = '/home/kumarv/khand035/Projects/MINT/river-depth-estimation/data/merged2/'
# fig_dir = '/home/kumarv/khand035/Projects/MINT/river-depth-estimation/results/versions/1/cross-sections_v3_merged2/'

data_dir = sys.argv[1]
cinfo = sys.argv[2]
base_dir = data_dir
prefix = cinfo[0:-4]
if '/' in prefix:
    prefix = prefix[prefix.rfind('/')+1:]
data_dir = data_dir + prefix + '/'
#
fig_dir  = data_dir

if os.path.isdir(fig_dir)==False:
    os.mkdir(fig_dir)
u = data_dir[0:-1].rfind('/')
data_name = data_dir[u+1:-1]

boxNum = data_name
# print(boxNum)
# os.system('rm -f ' + fig_dir + '*' + boxNum + '*.png')
# os.system('rm -f ' + fig_dir + '*' + boxNum + '*.npy')
# os.system('rm -f ' + fig_dir + '*' + boxNum + '*.csv')

dem_file = glob.glob(base_dir + boxNum + '/SRTM*.tif')[0]
# print(dem_file)
#print(boxNum)
flag = 0
mask_list = glob.glob(data_dir + 'O*.npy')
mask_list.sort()
datestr_list = []
datetim_list = []
datefmt_list = []
# print('Reading water extent maps...')
# print(mask_list)
if len(mask_list)==0:
    sys.exit()
for filename in mask_list:
    # print(filename)
    # u1 = filename.rfind('_')
    # u2 = filename.rfind('.')
    # datestr = filename[u1+1:u2]
    datestr = filename[-34:-26]
    # print(datestr)
    datefmt = datestr[0:4] + '-' + datestr[4:6] + '-' + datestr[6:8] + 'T' + '00' + ':' + '00' + ':' + '00'
    #print(datefmt)
    datenum = datetime.strptime(datefmt, '%Y-%m-%dT%H:%M:%S')
    datestr_list.append(datestr)
    datetim_list.append((datenum - datetime(1970,1,1)).days)
    datefmt_list.append(datefmt)
    mask_arr = np.load(filename)
    cols = mask_arr.shape[1]
    rows = mask_arr.shape[0]
    if flag==0:
        masks = np.zeros((rows,cols,len(mask_list)))
    masks[:,:,flag] = mask_arr
    flag = flag + 1

tmasks = masks.copy()
vmasks = tmasks<=1
tmasks[tmasks>1] = 0
smask = np.sum(tmasks,axis=2)
dmask = np.sum(vmasks,axis=2)
dmask[dmask==0] = 1
djrc = smask*1.0/dmask

#binarizing masks

temp = masks==3
masks[temp] = 0
temp = masks==2
masks[temp] = -1
masks = masks>0.5
masks = masks.astype(int)
masks[temp] = 2
# masks = masks.astype(int)
# print(masks.shape)

# print('Pruning water extent map to create river extent maps...')
ds = gdal.Open(dem_file)
delv = np.array(ds.GetRasterBand(1).ReadAsArray())
tdelv = delv.copy()
tdelv = tdelv.astype(float)
tdelv[tdelv==0] = np.nan

# djrc = np.mean(masks,axis=2)

# print('calculating center line')
cline_sum = djrc.copy()
cline_sum[:,:] = 0
bcount = 0
for i in range(100,20,-5):
    temp = djrc>i*1.0/100.0
    if np.sum(temp)==0:
        continue
    temp = ndimage.binary_fill_holes(temp)
    cline = skimage.morphology.skeletonize(temp)
    cline = cline.astype(int)
    cline_sum = cline_sum + cline*i
    bcount = bcount + i

# print(i)
cline_sum = cline_sum*1.0/bcount
fline = cline_sum>0.3
fline = ndimage.binary_fill_holes(fline)
cline = skimage.morphology.skeletonize(fline)

#---------------------------------need code to remove other water bodies------------------
pmasks = masks.copy()
# print(np.sum(pmasks==2))
# f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4)
# ax1.imshow(djrc)
# ax2.imshow(cline_sum)
# ax3.imshow(cline_sum>0.5)
# ax4.imshow(fline)
# plt.savefig(fig_dir + boxNum + '.png')
# sys.exit()


ref_extent = djrc>0.1
# pmasks = SelectLargestComponent(masks,cline)
# nmask = np.sum(pmasks==1,axis=2)*1.0
# dmask = np.sum(pmasks<=1,axis=2)*1.0
# pdjrc = np.divide(nmask,dmask)
# eth = 0.6
# emask = np.divide(nmask,dmask)>eth
# pmasks = RecoverLabels(pmasks,masks,pdjrc>eth)
# max_extent_ind = SelectMaximumExtent(np.divide(nmask,dmask)>0.1,pmasks)


# print('Extracting cross sections...')
csecs,ccord = PrepareCrossSections3(cline,dem_file)
num_secs = csecs.shape[2]
p = int(np.floor(0.2*num_secs))
# print(p,csecs.shape)
csecs = csecs[:,:,p:-1*p]
ccord = ccord[p:-1*p]
# print(csecs.shape)

# '''
f, (ax1, ax2, ax3, ax4,ax5,ax6) = plt.subplots(1,6, sharex=True, sharey=True,figsize=(10,5))
ax1.imshow(djrc)
ax1.set_title('Fraction Map')
ax2.imshow(cline)
ax2.set_title('Center Line')
temp = ref_extent.copy()
temp[np.sum(csecs[:,:,0:],axis=2)>0] = 2
ax3.imshow(temp)
ax3.set_title('Cross Sections')
ax4.imshow(tdelv)
ax4.set_title('Elevation')
ax5.imshow(djrc)
ax5.set_title('JRC Fractions')
ax6.imshow(ref_extent)
ax6.set_title('Maximum Extent')
plt.tight_layout()
f.savefig(fig_dir + 'Cat-' + boxNum + '-base.png')
# plt.show()
plt.close()

# sys.exit()
# '''


# print('Processing cross sections to approximate river bathymetry...')
elvp_arr = []
cls_min_arr = []
# lml_ind_arr = []
# lmr_ind_arr = []
labp_sec_arr = []
for j in range(0,csecs.shape[2]):
    cur_mask = ref_extent.copy()
    cur_sec = csecs[:,:,j].copy()
    cord = ccord[j]
    # if the cross section is empty...append default values
    if np.sum(cur_sec)==0 or cord[0]==500:
        cls_min_arr.append(-1)
        # lmr_ind_arr.append(-1)
        # lml_ind_arr.append(-1)
        elvp_arr.append(-1)
        continue

    # extract elevation and label profile for the given cross section
    labp,elvp,cp,rmask = GetCrossSectionProfiles4(cur_sec,cline,cur_mask,delv)
    # print(labp,len(labp),cp)
    if cp<0 or np.sum(elvp>0)==0 or np.sum(labp>1)>0:
        # print(cp,np.sum(elvp>0),np.sum(labp>1))
        cls_min_arr.append(-2)
        # lmr_ind_arr.append(-1)
        # lml_ind_arr.append(-1)
        elvp_arr.append(-2)
        continue


    # print(labp)
    labp = ExtractPiecewithCenterLine(labp,cp) # subset to only the piece with the center line
    if np.sum(labp)==0 or cp==-1:
        cls_min_arr.append(-3)
        # lmr_ind_arr.append(-1)
        # lml_ind_arr.append(-1)
        elvp_arr.append(-3)
        continue
    # print(labp)

    wts_inds = np.where(labp==1)[0]
    # applying iterative smoothing to elevation profile
    elvp_org = elvp.copy()
    sw = np.min([5,int(len(labp)*0.2)])
    emin = np.min(elvp)
    emax = np.max(elvp)
    elvp = IterativeConvolution(elvp,sw)
    elvp = np.convolve(elvp, np.ones((sw,))/sw, mode='same')
    elvp[0:int(sw/2)] = elvp_org[0:int(sw/2)]
    elvp[int(-sw/2):] = elvp_org[int(-sw/2):]


    # extract local maximas and minimas from the elevation profile
    # ielvp = elvp.copy()
    max_inds = argrelextrema(elvp, np.greater_equal,order=5)[0]
    min_inds = argrelextrema(elvp, np.less_equal,order=5)[0]


    # scoring each local minima based on how much
    # percentage of label profile falls into the local bowl
    score = np.zeros((len(min_inds),))
    for cls_min in range(0,len(min_inds)):

        cur_dif = max_inds - min_inds[cls_min]
        cur_dif[cur_dif>0] = np.min(cur_dif)-1
        lml_ind = np.argmax(cur_dif)
        cur_dif = max_inds - min_inds[cls_min]
        cur_dif[cur_dif<0] = np.max(cur_dif)+1
        lmr_ind = np.argmin(cur_dif)
        score[cls_min] = np.sum(labp[max_inds[lml_ind]:max_inds[lmr_ind]]==1)*1.0/np.sum(labp==1)


    # calculating the two local maximas for the selected local minima
    cls_min = np.argsort(score)[-1]
    cls_min_arr.append(min_inds[cls_min])
    elvp_arr.append(elvp)
    labp_sec_arr.append(labp)


    # calculating score for this cross section
    emin = min([elvp[wts_inds[0]],elvp[wts_inds[-1]]])
    earr = elvp[wts_inds[0]:wts_inds[-1]]
    edif = np.round(np.sum(emin-earr),3)
    if edif<=0:
        edif = 'M' + str(abs(edif))
    else:
        edif = 'P' + str(abs(edif))

    fig, (ax1,ax3) = plt.subplots(1,2,figsize=(15,5))
    ax1.plot(elvp,'--b',linewidth=1.5)
    ax1.plot(elvp_org,'--k',linewidth=0.5)
    ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim([elvp[min_inds[cls_min]]-5,emax])
    ax1.plot([wts_inds[0],wts_inds[-1]],[elvp[wts_inds[0]],elvp[wts_inds[-1]]],'-r',linewidth=2)
    ax1.plot(cp,elvp[cp],'*k',linewidth=2)

    ax3.plot()
    cur_mask[cur_sec==1] = 2
    cur_mask = cur_mask.astype(int)
    cur_mask[cur_mask>0] = np.max(rmask)
    cur_mask[rmask>0] = rmask[rmask>0]
    ax3.imshow(cur_mask)
    plt.tight_layout()
    fig.savefig(fig_dir + str(edif) + '_Cat-' + boxNum + '-sec-' + str(j) + '.png')
    # plt.show()
    plt.close()
    np.save(fig_dir + 'earr-' + boxNum + '-sec-' + str(j),earr)

    # '''
# print(cls_min_arr)
# sys.exit()
# sys.exit()
# print('Creating calibration files...')
# fid = open(fig_dir + data_name + '-viz.csv','w')

# plt.close('all')
# print(datestr_list)
for c in range(0,csecs.shape[2]):
    #print(c)
    # lml_ind = lml_ind_arr[c]
    # lmr_ind = lmr_ind_arr[c]
    cls_min = cls_min_arr[c]
    ielvp = elvp_arr[c]
    if cls_min<0:
        # print('issue with cls_min')
        continue

    fid2 = open(fig_dir + data_name + '-sec-' + str(c) + '.csv','w')
    fid3 = open(fig_dir + data_name + '-vizlocal-' + str(c) + '.csv','w')
    fid = open(fig_dir + data_name + '-viz-' + str(c) + '.csv','w')

    cur_sec = csecs[:,:,c].copy()
    np.save(fig_dir + data_name + '-sec-' + str(c),cur_sec)
    cord = ccord[c]
    sdate = (datetime(2015,1,1) - datetime(1970,1,1)).days
    edate = (datetime(2019,12,31) - datetime(1970,1,1)).days
    w_full = np.zeros((edate-sdate+1,))
    da_full = np.zeros((edate-sdate+1,))
    fullx = np.arange(sdate,edate+1)
    # print(fullx)
    # print(np.where(fullx==datetim_list[0])[0][0])
    sind = np.where(fullx==datetim_list[0])[0][0]
    eind = np.where(fullx==datetim_list[-1])[0][0]

    numt = pmasks.shape[2]
    # wts = np.zeros((numt,))

    for i in range(0,numt):
        #print(c,i)
        cur_mask = pmasks[:,:,i].copy()
        labp,elvp,cp,rmask = GetCrossSectionProfiles4(cur_sec,cline,cur_mask,delv)
        # print(labp)
        # fig, (ax1,ax3) = plt.subplots(1,2,figsize=(15,5))
        # ax1.plot(ielvp,'--b',linewidth=1.5)
        # # ax1.plot(elvp_org,'--k',linewidth=0.5)
        # ax1.set_ylabel('Elevation (m)')
        # ax1.set_ylim([ielvp[cls_min]-5,np.max(ielvp)])
        #
        # tielvp = ielvp.copy()
        # tielvp = tielvp.astype(float)
        # tielvp[labp!=0] = np.nan
        # ax1.plot(tielvp,'-r',linewidth=2)
        #
        # tielvp = ielvp.copy()
        # tielvp = tielvp.astype(float)
        # tielvp[labp!=1] = np.nan
        # ax1.plot(tielvp,'-g',linewidth=2)
        #
        # tielvp = ielvp.copy()
        # tielvp = tielvp.astype(float)
        # tielvp[labp!=2] = np.nan
        # ax1.plot(tielvp,'-m',linewidth=2)
        #
        #
        # plt.tight_layout()
        # # fig.savefig(fig_dir + str(edif) + '_Cat-' + boxNum + '-sec-' + str(j) + '.png')
        # plt.show()
        # plt.close()

        if np.sum(labp>1)>0:
            fid2.write(datefmt_list[i] + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(cp) + ',' + str(cord[0]) + ',' + str(cord[1]) + '\n')
            continue

        labp = ExtractPiecewithCenterLine(labp,cp) # subset to only the piece with the center line


        wts_inds = np.where(labp==1)[0]
        if wts_inds.shape[0]==0:
            fid2.write(datefmt_list[i] + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(cp) + ',' + str(cord[0]) + ',' + str(cord[1]) + '\n')
            # wts[i] = 0
        else:


            ledge = wts_inds[0]
            redge = wts_inds[-1]
            if ledge>cls_min or redge<cls_min:
                redge = -1
                ledge = -1

            #print(wts_inds,ledge,redge)
            if ledge>=0 and redge>=0:
                ldif = ielvp[ledge]-ielvp[cls_min]
                rdif = ielvp[redge]-ielvp[cls_min]
                adif = (ldif + rdif)*0.5
                #date,width,edif_left,edif_right,edif_avg,ele_center,index_left,index_center,index_right,index_cline,lat,lon
                fid2.write(datefmt_list[i] + ',' + str(redge-ledge+1) + ',' + str(round(ielvp[ledge]-ielvp[cls_min],2)) + ',' + str(round(ielvp[redge]-ielvp[cls_min],2)) + ',' + str(round(adif,2)) + ',' + str(round(ielvp[cls_min],2)) + ',' + str(ledge) + ',' + str(cls_min) + ',' + str(redge) + ',' + str(cp) + ',' + str(cord[0]) + ',' + str(cord[1]) + '\n')
            else:
                fid2.write(datefmt_list[i] + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(round(ielvp[cls_min],2)) + ',' + str(ledge) + ',' + str(cls_min) + ',' + str(redge) + ',' + str(cp) + ',' + str(cord[0]) + ',' + str(cord[1]) + '\n')

            # wts[i] = np.sum(labp==1)
    fid2.close()
    da,dl,dr,w,xticklabels_all,flag = CreateTimeSeries(fig_dir,boxNum,str(c))
    da = da.astype(float)
    # print(da)
    # if w==-1:
    #     print('bad w')
    #     continue
    bad_inds = np.isnan(w)
    # print(w)

    temp = []
    xticklabels = []
    for z in range(0,len(datetim_list)):
        if np.isnan(w[z])==False:
            temp.append(datetim_list[z])
            xticklabels.append(xticklabels_all[z])
    if len(temp)<2:
        continue
    # print(len(temp)
    sind = np.where(fullx==temp[0])[0][0]
    eind = np.where(fullx==temp[-1])[0][0]

    w = w[bad_inds==False]
    # print(temp.shape)
    f = interp1d(temp,w,kind='linear')
    newx = np.arange(temp[0],temp[-1]+1)
    newy = f(newx)
    w_full[0:sind] = w[0]
    w_full[eind:] = w[-1]
    w_full[sind:eind+1] = newy


    da = da[bad_inds==False]
    f = interp1d(temp,da,kind='linear')
    newx = np.arange(temp[0],temp[-1]+1)
    newy = f(newx)
    da_full[0:sind] = da[0]
    da_full[eind:] = da[-1]
    da_full[sind:eind+1] = newy

    # print(np.arange(len(xticklabels)))

    tf,ax = plt.subplots()
    ax.plot(w*10,'-*r')
    ax.set_ylabel('River Width (m)',color='red')
    ax.set_ylim([(np.nanmin(w)-1)*10,(np.nanmax(w)+1)*10])
    plt.xticks(np.arange(len(xticklabels)),(xticklabels))
    plt.xticks(rotation=90)
    ax.set_xlabel('Date')
    axt = ax.twinx()
    axt.plot(da,'-sb')
    # axt.set_xticklabels(np.arange(len(xticklabels)),xticklabels)
    axt.set_ylabel('River depth (m)',color='blue')

    # plt.tight_layout()
    # plt.show()
    plt.savefig(fig_dir + 'ts' + '-' + boxNum + '-wts-' + str(c) + '.png')
    plt.close()
    # plt.plot(temp,w,'*b')
    # plt.plot(fullx,w_full,'.r')
    # plt.show()
    # fig.savefig(data_dir + 'Cat' + '-' + boxNum + '-wts-' + str(c) + '.png')

    for i in range(0,len(w_full)):
        new_date = datetime(1970,1,1) + timedelta(int(fullx[i]))
        new_date_str = new_date.strftime('%Y-%m-%d')
        if fullx[i] in temp:
            isorg = 1
        else:
            isorg = 0
        fid.write(new_date_str + ',' + str(round(w_full[i],2)) + ',' + str(round(da_full[i],2)) + ',' + str(isorg) + ', POINT (' + str(cord[0]) + ' ' + str(cord[1]) + ') \n')
        fid3.write(new_date_str + ',' + str(round(w_full[i],2)) + ',' + str(round(da_full[i],2)) + ',' + str(isorg) + ',' + str(cord[0]) + ',' + str(cord[1]) + '\n')
    fid3.close()
    fid.close()
# print('rm -f ' + boxNum + '_wgs84.tif')
os.system('rm -f ' + boxNum + '_wgs84.tif')
os.system('rm -f ' + boxNum + '.tif')
os.system('touch ' + fig_dir + 'done_' + boxNum + '.txt')
