# try the folloqing as input - area around NY region
# python DownloadRNNData.py lat_min lon_min lat_max lon_max sdate edate out_path
import numpy as np
import sys,os
import descarteslabs as dl
import pyproj
import matplotlib.pyplot as plt
import gdal,ogr,osr
raster_client = dl.Raster()


def search_AOI(box):
#date time must have standard format: 2018-07-01, NOT 2018-7-1
    scenes, geoctx = dl.scenes.search(box,
        products=["sentinel-1:GRD"],
        #products=['airbus:oneatlas:phr:v2'],
        start_datetime="2018-08-01",
        end_datetime="2018-10-01")
        
    return scenes, geoctx





box_file = sys.argv[1]
sdate = sys.argv[2]
edate = sys.argv[3]
out_base = sys.argv[4]
prefix = sys.argv[5]

# opening the shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()

 
for feature in cdl:

    # get boundary information of the partition
    geom = feature.GetGeometryRef()
    #geom.Transform(coordTrans)
    minX, maxX, minY, maxY = geom.GetEnvelope() #bounding box of the box
    curID = feature.GetField('BID')
    lat_min = minY
    lat_max = maxY
    lon_min = minX
    lon_max = maxX
    print lat_min,lon_min,lat_max,lon_max
    out_path = out_base + prefix + '-' + str(curID) + '/'
    if os.path.isdir(out_path)==False:
        os.mkdir(out_path)
   
    aoi_geometry = {'type': 'Polygon',
                'coordinates': (((lon_min, lat_min),
                                 (lon_max, lat_min),
                                 (lon_max, lat_max),
                                 (lon_min, lat_max),
                                 (lon_min, lat_min)),)}


    scenes, geoctx = dl.scenes.search(aoi_geometry, products=["sentinel-1:GRD"],start_datetime=sdate,end_datetime=edate)
    print('Number of scenes found: ' + str(len(scenes)))
    for i in range(0,len(scenes)):
        curid = scenes[i].properties.id
        curname = scenes[i].properties.identifier
        print curname
        raster_file=raster_client.raster(inputs=curid,bands=['vv'],data_type='Byte',align_pixels=True,cutline=aoi_geometry,save=True,outfile_basename=out_path + curname + '.VV',output_format='GTiff',resolution=5)
        raster_file=raster_client.raster(inputs=curid,bands=['vh'],data_type='Byte',align_pixels=True,cutline=aoi_geometry,save=True,outfile_basename=out_path + curname + '.VH',output_format='GTiff',resolution=5)



        


