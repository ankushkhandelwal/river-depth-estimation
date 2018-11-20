import os
import sys
import ogr,osr
import pandas

box_file = sys.argv[1] # full path of the shapefile containing boxes
sdate = sys.argv[2] # starting date in yyyymmdd format
edate = sys.argv[3] # ending date in yyyymmdd format
cfile = sys.argv[4] # full path to the credentials file

# reading the first set of credentials
cf = pandas.read_csv(cfile,names=['username','password'])
username = cf['username'][0]
password = cf['password'][0]

# opening the shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
cds = driver.Open(box_file, 0)
cdl = cds.GetLayer()

# preparding cordinate system transformation. This will update in later versions
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(20136)

outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(4326)

coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)


# looping through boxes
lfa = open('file-list-' + 'all' + '.txt', 'w') # file to store scene information from all boxes
for feature in cdl:

    # get boundary information of the partition
    geom = feature.GetGeometryRef()
    geom.Transform(coordTrans)
    minX, maxX, minY, maxY = geom.GetEnvelope()
    curID = feature.GetField('BID')
    ilat_min = minY
    ilat_max = maxY
    ilon_min = minX
    ilon_max = maxX
    bbox_str = str(ilon_min) + '%20' + str(ilat_min) + ',' + str(ilon_max) + '%20' + str(ilat_min) + ',' + str(ilon_max) + '%20' + str(ilat_max) + ',' + str(ilon_min) + '%20' + str(ilat_max) + ',' + str(ilon_min) + '%20' + str(ilat_min)
    query = '(platformname:Sentinel-1%20AND%20producttype:GRD%20AND%20beginposition:%5b' + sdate + 'T00:00:00.000Z%20TO%20' + edate + 'T00:00:00.000Z%5d%20AND%20footprint:%22Intersects(POLYGON((' + bbox_str + ')))%22)'
    os.system('curl -s -u ' + username + ':' + password + ' "https://scihub.copernicus.eu/dhus/search?q=' + query + '" > code_test1.txt')

    #extracting number of scenes available for the current feature
    with open('code_test1.txt') as f:
        for line in f:
            if '<opensearch:totalResults>' in line:
                sind = line.find('>')
                eind = line.find('/')
                num_scenes = int(line[sind+1:eind-1])
                break
    f.close()
    print 'Total number of scenes in the result: ' + str(num_scenes)


    lf = open('file-list-' + str(curID) + '.txt','w')
    for i in range(0,num_scenes,100):
        new_query = query + '&rows=100&start=' + str(i)
        os.system('curl -s -u ' + username + ':' + password + ' "https://scihub.copernicus.eu/dhus/search?q=' + new_query + '" > code_test1.txt')
        with open('code_test1.txt') as f:
            for line in f:
                if '<link href=' in line:
                    sind = line.find('"')
                    eind = line.rfind(')')
                    blink = line[sind + 1:eind+1]
                    tlink =  blink + '/Online/$value'
                    dlink = blink + '/\$value'
                    os.system('curl -s -u ' + username + ':' + password + ' ' + '"' + tlink + '" > status.txt')
                    with open('status.txt') as sf:
                        for sline in sf:
                            if 'true' in sline:
                                print dlink
                                lf.write(dlink + ',')
                                lfa.write(dlink + ',')
                                #os.system('wget --content-disposition --continue --user=ankushumn --password=michinel21c ' + '"' + dlink + '"')

                if '<str name="filename">' in line:
                    sind = line.find('>')
                    eind = line.find('/')
                    fname = line[sind+1:eind-1]
                    lf.write(fname + '\n')
                    lfa.write(fname + '\n')

        f.close()
    lf.close()
lfa.close()
