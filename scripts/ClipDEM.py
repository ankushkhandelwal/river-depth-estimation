import os
import sys
import gdal,ogr,osr
import pandas

box_file =sys.argv[1] # full path of the shapefile containing boxes
dem_file = sys.argv[2] # full path to dem file
cfile = sys.argv[3] # file that contains the credentials
out_base = sys.argv[4] # base path where folders for individual boxes would be made
prefix = sys.argv[5]

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

cds.Destroy()

dem_name = dem_file[dem_file.rfind('/')+1:]
Raster = gdal.Open(dem_file, 1)
Projection = Raster.GetProjectionRef()
#os.system('gdalwarp -q -overwrite -tps -tr 10 10 -t_srs EPSG:20136 ' + tif_dir + tif_file + ' ' + tif_dir + 'r' + tif_file)

driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()
# clipping the tif file for each box
for feature in cdl:
    geom = feature.GetGeometryRef()
    curID = feature.GetField('BID')
    minX, maxX, minY, maxY = geom.GetEnvelope()
    print minX, maxX, minY, maxY
    out_dir = out_base + prefix + '-' + str(curID) + '/'
    OutTileName = out_dir + 'dem.tiff'
    print OutTileName
    #clip command
    OutTile = gdal.Warp(OutTileName, Raster, format=RasterFormat, outputBounds=[minX, minY, maxX, maxY],
                        xRes=xPixelRes, yRes=yPixelRes, dstSRS=Projection, resampleAlg=gdal.GRA_NearestNeighbour)

Raster = None
cds.Destroy()


