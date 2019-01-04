import os
import sys
import gdal,ogr,osr
import pandas

box_file =sys.argv[1] # full path of the shapefile containing boxes
data_dir = sys.argv[2] # full path of the folder that would contain the zip files
cfile = sys.argv[3] # file that contains the credentials
out_base = sys.argv[4] # base path where folders for individual boxes would be made
prefix = sys.argv[5]
dem_file = sys.argv[6]
# creating base directory if does not already exists
if os.path.isdir(out_base)==False:
    os.mkdir(out_base)

# reading the first set of credentials 
cf = pandas.read_csv(cfile,names=['username','password'])
username = cf['username'][0]
password = cf['password'][0]

# constants
RasterFormat = 'GTiff'
VectorFormat = 'ESRI Shapefile'

xPixelRes = 0.00027777778/3.0
yPixelRes = -0.00027777778/3.0

# reading the list of all scenes as a data-frame
df = pandas.read_csv('file-list-all.txt',names=['link','name'])
dfu = df.drop_duplicates()

# opening the shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()

# get boundary information of each box
boxes = []
for feature in cdl:
    geom = feature.GetGeometryRef()
    boxes.append(geom)
    BID = feature.GetField('BID')
    out_dir = out_base + prefix + '-' + str(BID) + '/'
    if os.path.isdir(out_dir)==False:
        os.mkdir(out_dir)


# clipping each scene
for index,scene in dfu.iterrows():
    dlink = scene['link']
    zip_file = scene['name'][0:-5] + '.zip'
    print zip_file
    # checking if the zip file already exists. This part will not get used in pegasus workflow
    if os.path.isfile(data_dir + zip_file) == False:
        #os.system('wget --directory-prefix=' + data_dir + ' --content-disposition --continue --user=' + username + ' --password=' + password + ' ' + '"' + dlink + '"')
        os.system('wget --directory-prefix=' + data_dir + ' "' + dlink + '"')
    
    continue    
    # checking if the zip file is already unzipped. This part will always get called in pegasus workflow    
    if os.path.isfile(data_dir + zip_file) == False:
        continue

    if os.path.isdir(data_dir + zip_file[0:-4] + '.SAFE') == False:
        os.system('unzip -q ' + data_dir + zip_file + ' -d ' + data_dir)


    # clipping both vv and vh tif files for each box
    tif_dir = data_dir + zip_file[0:-4]+ '.SAFE/measurement/'
    for tif_file in os.listdir(tif_dir):
        if tif_file.endswith('tiff')==False or tif_file[0]=='r':
            continue

        #adding projection to the tif file. This part will be updated in later versions
        #os.system('gdalwarp -q -overwrite -tps -tr 10 10 -t_srs EPSG:20136 ' + tif_dir + tif_file + ' ' + tif_dir + 'r' + tif_file)
        print('gdalwarp -t_srs ' + dem_file + ' ' + tif_dir + tif_file + ' ' + tif_dir + 'r' + tif_file)
        
        os.system('gdalwarp -t_srs ' + dem_file + ' ' + tif_dir + tif_file + ' ' + tif_dir + 'r' + tif_file)

        Raster = gdal.Open(tif_dir + 'r' + tif_file, 1)
        Projection = Raster.GetProjectionRef()

        driver = ogr.GetDriverByName("ESRI Shapefile")
        cds = driver.Open(box_file, 0)
        cdl = cds.GetLayer()
        # clipping the tif file for each box
        for feature in cdl:
            geom = feature.GetGeometryRef()
            curID = feature.GetField('BID')
            minX, maxX, minY, maxY = geom.GetEnvelope()
          

            out_dir = out_base + prefix + '-' + str(curID) + '/'
            fid = open(out_dir + 'bbox.txt','w')
            fid.write(str(minY) + ' ' + str(minX) + ' ' + str(maxY) + ' ' + str(maxX) + '\n')
            fid.close()
            DemRaster = gdal.Open(out_dir + '/dem.tiff', 1)
            band = DemRaster.GetRasterBand(1)
            width = band.XSize
            height = band.YSize
            DemRaster = None
            OutTileName = out_dir + 'r' + tif_file
            print OutTileName
            #clip command
            OutTile = gdal.Warp(OutTileName, Raster, format=RasterFormat, outputBounds=[minX, minY, maxX, maxY],
                                width=width, height=height,dstSRS=Projection, resampleAlg=gdal.GRA_NearestNeighbour)


        Raster = None
        cds.Destroy()
    # deleting the original unzipped folder and zip file to avoid space usage
    os.system('rm -rf ' + data_dir + zip_file[0:-4]+ '.SAFE/')
    #os.system('rm -rf ' + data_dir + zip_file)


