import os
import sys
import gdal,ogr,osr
import pandas

box_file =sys.argv[1] # full path of the shapefile containing boxes
dem_file = sys.argv[2] # full path to dem file
cfile = sys.argv[3] # file that contains the credentials
out_base = sys.argv[4] # base path where folders for individual boxes would be made
prefix = sys.argv[5]
basin_file = sys.argv[6]
basin_id = sys.argv[7]
num_boxes = int(sys.argv[8])
out_file = sys.argv[9]

# creating base directory if does not already exists
if os.path.isdir(out_base)==False:
    os.mkdir(out_base)

# reading the first set of credentials 
#cf = pandas.read_csv(cfile,names=['username','password'])
#username = cf['username'][0]
#password = cf['password'][0]

# constants
RasterFormat = 'GTiff'
VectorFormat = 'ESRI Shapefile'
xPixelRes = 0.00027777778/3.0
yPixelRes = -0.00027777778/3.0

outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(4326)

dem_name = dem_file[dem_file.rfind('/')+1:]
Raster = gdal.Open(dem_file, 1)
Projection = Raster.GetProjectionRef()
#os.system('gdalwarp -q -overwrite -tps -tr 10 10 -t_srs EPSG:20136 ' + tif_dir + tif_file + ' ' + tif_dir + 'r' + tif_file)

driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()

bds = driver.Open(basin_file, 0)
bdl = bds.GetLayer()
bdl.SetAttributeFilter("catNum = " + basin_id)
print 'Number of selected sub-basins: ' + str(bdl.GetFeatureCount())

for bfeature in bdl:
    bgeom = bfeature.GetGeometryRef()

cdl.SetSpatialFilter(bgeom)
print 'Number of boxes in the sub-basin: ' + str(cdl.GetFeatureCount())

ctr = 0
# clipping the tif file for each box
for feature in cdl:
    #geom = feature.GetGeometryRef()
    curID = feature.GetField('BID')
    #minX, maxX, minY, maxY = geom.GetEnvelope()
    #print minX, maxX, minY, maxY
    out_dir = out_base + prefix + '-' + str(curID) + '/'
    for tifname in os.listdir(out_dir):
        if tifname.endswith('tif')==True:
            break
    os.system('gdaltindex ' + out_dir + 'bbox.shp ' + out_dir + tifname)
    bds = driver.Open(out_dir + 'bbox.shp',0)
    bdl = bds.GetLayer()
    for bfeature in bdl:
	a = 1
    bgeom = bfeature.GetGeometryRef()
    BaseRaster = gdal.Open(out_dir + tifname, 1)
    inSpatialRef = osr.SpatialReference(wkt=BaseRaster.GetProjection())	
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    bgeom.Transform(coordTrans)
    minX, maxX, minY, maxY = bgeom.GetEnvelope()
    bds.Destroy()

    band = BaseRaster.GetRasterBand(1)
    width = band.XSize
    height = band.YSize
    BaseRaster = None
            
    OutTileName = out_dir + out_file
    print OutTileName
    #clip command
    #OutTile = gdal.Warp(OutTileName, Raster, format=RasterFormat, outputBounds=[minX, minY, maxX, maxY],
    #                    xRes=xPixelRes, yRes=yPixelRes, dstSRS=Projection, resampleAlg=gdal.GRA_NearestNeighbour)
    OutTile = gdal.Warp(OutTileName, Raster, format=RasterFormat, outputBounds=[minX, minY, maxX, maxY],
                                width=width, height=height,dstSRS=Projection, resampleAlg=gdal.GRA_NearestNeighbour)
    
    ctr = ctr + 1
    if ctr==num_boxes:
        break

Raster = None
cds.Destroy()


