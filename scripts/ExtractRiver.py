import os
import sys
import numpy as np
import skimage
import skimage.measure
import skimage.morphology
import glob
import gdal
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import matplotlib.pyplot as plt

# return the mask with only the largest connected component
def SelectLargestComponent(masks):
    rows,cols,numt = masks.shape
    new_masks = np.zeros((rows,cols,numt))
    
    for i in range(0,numt):
        cur_mask = masks[:,:,i]
        cur_labels = skimage.measure.label(cur_mask,background=0)
#         if i ==0:
#             f, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True,figsize=(15,10))
#             ax1.imshow(cur_mask)
#             ax2.imshow(cur_labels)
            
#             temp = np.bincount(cur_labels.flatten())
#             print temp
        
        if np.max(cur_labels)==0:
            continue
            
        lcc = np.argmax(np.bincount(cur_labels.flatten())[1:])
        new_mask = cur_labels==lcc+1
        new_masks[:,:,i] = new_mask
        
    return new_masks

# returns the lat,lon cordinates of a pixel
def pix2coord(x,y,image):
    ds = gdal.Open(image,0)
    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()

    """Returns global coordinates from pixel x, y coords"""
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

# returns an array of masks containing cross sections
# also returns the lat,lon array of locations on the center-line 
# that was used to create the cross-section
def PrepareCrossSections2(image,geobase):
    sr = 10 # start row
    jump = 10
    sw = 3 # section winow width
    row1, col1 = image.shape
    linds = np.where(image==1)
    rinds = linds[0]
    cinds = linds[1]
    pdist = np.multiply(rinds,rinds) + np.multiply(cinds,cinds)
    pargs = np.argsort(pdist)
    
    temp = np.arange(sr,pargs.shape[0],jump)
    cross_image = np.zeros((row1,col1,temp.shape[0])) # store each cross section separately
    coord_info = np.zeros((temp.shape[0],2)) # store each cross section separately
    
    ctr = 0
    for p in range(sr,pargs.shape[0],jump):
        cur_image = np.zeros((row1,col1))
        i = rinds[pargs[p]]
        j = cinds[pargs[p]]
        clat,clon = pix2coord(i,j,geobase)
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

# returns the elevation and label profile
# for a given cross section and date
def GetCrossSectionProfiles(csec,cline,lmask,delv):
    linds = np.where(csec==1)
    rinds = linds[0]
    cinds = linds[1]
    pdist = np.multiply(rinds,rinds) + np.multiply(cinds,cinds)
    pargs = np.argsort(pdist)
    
    temp = np.arange(0,pargs.shape[0])
    labp = np.zeros((temp.shape[0],))
    elep = np.zeros((temp.shape[0],))
    ctr = 0
    for p in range(0,temp.shape[0]):
        i = rinds[pargs[p]]
        j = cinds[pargs[p]]
        labp[p] = lmask[i,j]
        elep[p] = delv[i,j]
        if cline[i,j]==1:
            labp[p] = 2
        
    return labp,elep



#------------------------------main script----------------------

data_dir = sys.argv[1]
out_file = sys.argv[2] # path of csv file to store the area
isGEE = sys.argv[3]

# extracting file names and dates
if isGEE=='1':
    # extracting the names
    ext = 'tif'
    fnames = []
    for fname in os.listdir(data_dir):
        if fname.endswith('tif')==False or fname[0:2]=='20':
            continue
        fnames.append(fname[17:48])

elif isGEE=='2':
    
    # extracting the names
    ext = 'tif'
    fnames = []
    for fname in os.listdir(data_dir):
        if fname.endswith('tif')==False or fname[0]=='r' or fname[0:2]=='20':
            continue
        fnames.append(fname[17:-26])

else:
    # extracting the names
    ext = 'tiff'
    fnames = []
    for fname in os.listdir(data_dir):
        if fname.endswith('tiff')==False or fname[0]!='r' or fname[0:2]=='20':
            continue
        fnames.append(fname[15:-9])

# keeping only unique entries and sorting them
fnames = list(set(fnames))
fnames.sort()
#print fnames
start_date = fnames[0][0:4] + '-' + fnames[0][4:6] + '-' + fnames[0][6:8] + 'T' + fnames[0][9:11] + ':' + fnames[0][11:13] + ':' + fnames[0][13:15]
end_date = fnames[-1][0:4] + '-' + fnames[-1][4:6] + '-' + fnames[-1][6:8] + 'T' + fnames[-1][9:11] + ':' + fnames[-1][11:13] + ':' + fnames[-1][13:15]

# loading each mask into an array of masks

for i in range(0,len(fnames)):
    fname = fnames[i]
    filename = data_dir + fname + '.' + ext
    ds = gdal.Open(filename)
    numchannels = ds.RasterCount
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    mask_arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    if i ==0:
            masks = np.zeros((rows,cols,len(fnames)))
    masks[:,:,i] = mask_arr
    
    

'''
mask_list = glob.glob(data_dir + '20*.tif')
for filename in glob.glob(data_dir + '20*.tif'):
    #print filename
    ds = gdal.Open(filename)
    numchannels = ds.RasterCount
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    mask_arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    if flag==0:
        masks = np.zeros((rows,cols,len(mask_list)))
    masks[:,:,flag] = mask_arr
    flag = flag + 1
'''

pmasks = SelectLargestComponent(masks) # pruned masks
emask = np.sum(pmasks,axis=2)>0 # max extent map
cline = skimage.morphology.skeletonize(emask)# center line mask

#preparing cross sections
csecs,ccord = PrepareCrossSections2(cline,data_dir + 'dem.tiff')


'''
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True,figsize=(15,10))
ax1.imshow(emask)
ax2.imshow(cline)
ax3.imshow(np.sum(csecs,axis=2))
plt.show()
'''

# reading elevation data
ds = gdal.Open(data_dir + 'dem.tiff')
delv = np.array(ds.GetRasterBand(1).ReadAsArray())

# selecting a specific cross section
cur_sec = csecs[:,:,0]


f = open(data_dir + 'calibration-series.txt','w')
for i in range(0,pmasks.shape[2]):
    cur_mask = pmasks[:,:,i]
    labp,elvp = GetCrossSectionProfiles(cur_sec,cline,cur_mask,delv)
    datestr = fname[0:4] + '-' + fname[4:6] + '-' + fname[6:8] + 'T' + fname[9:11] + ':' + fname[11:13] + ':' + fname[13:15]
    f.write(datestr + ',' + str(np.sum(labp>0)) + '\n')




'''
for i in range(0,csecs.shape[2]):
    cur_dem = delv.copy()
    cur_sec = csecs[:,:,i]
    cur_dem[cur_sec==0] = 0
    
    f, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True,figsize=(15,10))
    ax1.imshow(cur_sec)
    ax2.imshow(cur_dem)
    plt.show()
'''   
    


                         
    
'''    
    if mask[0:2] != '20':
        continue
        
        ds = gdal.Open(data_dir + mask)
        numchannels = ds.RasterCount
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        mask_arr = np.array(ds.GetRasterBand(1).ReadAsArray())
        if flag==0:
            mask_sum = mask_arr
            flag = 1
        else:
            mask_sum = mask_sum + mask_arr

        f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True,figsize=(15,10))
        ax1.imshow(mask_arr)
        ax2.imshow(vvs_arr)
        ax2.set_title(mask[0:8])
        ax3.imshow(vhs_arr)
'''
        