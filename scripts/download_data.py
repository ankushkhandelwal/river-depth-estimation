import numpy as np
import sys,os
import descarteslabs as dl
import pyproj
import gdal,ogr,osr
raster_client = dl.Raster()
import glob

def search_AOI(box):
#date time must have standard format: 2018-07-01, NOT 2018-7-1
    scenes, geoctx = dl.scenes.search(box,
        products=["sentinel-1:GRD"],
        #products=['airbus:oneatlas:phr:v2'],
        start_datetime="2018-08-01",
        end_datetime="2018-10-01")

    return scenes, geoctx


sdate = sys.argv[1]
edate = sys.argv[2]
out_base = sys.argv[3]
cinfo = sys.argv[4]
buf_dist = float(sys.argv[5])
prefix = cinfo[0:-4]
if '/' in prefix:
    prefix = prefix[prefix.rfind('/')+1:]

with open(cinfo) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
inpID = content[0]

temp = inpID.split(',')
lat = float(temp[0])
lon = float(temp[1])

point = ogr.Geometry(ogr.wkbPoint)
point.AddPoint(lon,lat)
box = point.Buffer(buf_dist)
minX, maxX, minY, maxY = box.GetEnvelope() #bounding box of the box

lat_min = minY
lat_max = maxY
lon_min = minX
lon_max = maxX

print(lat_min,lon_min,lat_max,lon_max)
out_path = out_base + prefix + '/'
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
    print(curname)
    if os.path.isfile(out_path + curname + '.tif')==True:
        continue
    raster_file=raster_client.raster(inputs=curid,bands=['vv','vh'],data_type='Byte',align_pixels=True,cutline=aoi_geometry,save=True,outfile_basename=out_path + curname,output_format='GTiff',resolution=10)



scenes, geoctx = dl.scenes.search(aoi_geometry, products=["sentinel-2:L1C"],start_datetime=sdate,end_datetime=edate)
print('Number of scenes found: ' + str(len(scenes)))
for i in range(0,len(scenes)):
    curid = scenes[i].properties.id
    curname = scenes[i].properties.identifier
    print(curname)
    if os.path.isfile(out_path + curname + '.tif')==True:
        continue
    raster_file=raster_client.raster(inputs=curid,bands=["coastal-aerosol", "blue", "green", "red", "red-edge","red-edge-2","red-edge-3","nir","red-edge-4","water-vapor","cirrus","swir1","swir2","bright-mask","cirrus-cloud-mask","cloud-mask","opaque-cloud-mask"],data_type='UInt16',align_pixels=True,cutline=aoi_geometry,save=True,outfile_basename=out_path + curname,output_format='GTiff',resolution=10)

scenes, geoctx = dl.scenes.search(aoi_geometry, products=["srtm:GL1003"],start_datetime='1999-01-01',end_datetime='2001-01-01')
print('Number of scenes found: ' + str(len(scenes)))
for i in range(0,len(scenes)):
    curid = scenes[i].properties.id
    # print curid
    # curname = scenes[i].properties.identifier
    curname = 'SRTM'
    print(curname)
    if len(glob.glob(out_path + curname + '*.tif'))>0:
        continue
    raster_file=raster_client.raster(inputs=curid,bands=["altitude","aspect","slope"],data_type='UInt16',align_pixels=True,cutline=aoi_geometry,save=True,outfile_basename=out_path + curname,output_format='GTiff',resolution=10)


os.system('chmod -R 775 ' + out_path)
print('done')
