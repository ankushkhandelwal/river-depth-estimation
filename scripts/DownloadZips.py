import os
import sys
import pandas


sind = int(sys.argv[1])
eind = int(sys.argv[2])
data_dir = sys.argv[3]

# reading the list of all scenes as a data-frame
df = pandas.read_csv('file-list-all.txt',names=['link','name'])
dfu = df.drop_duplicates()

ctr = 0
for index,scene in dfu.iterrows():
    dlink = scene['link']
    zip_file = scene['name'][0:-5] + '.zip'
    if os.path.isfile(data_dir + zip_file)==True:
        ctr = ctr + 1
        continue
    if ctr>=sind and ctr<=eind:
        os.system('wget --directory-prefix=' + data_dir + ' "' + dlink + '"')
    ctr = ctr + 1

