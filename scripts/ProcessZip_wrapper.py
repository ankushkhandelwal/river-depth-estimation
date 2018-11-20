import os
import sys
import ProcessZip


data_dir = sys.argv[1] # path of box folder where clipped files are stored
out_file = sys.argv[2] # path of csv file to store the area

# extracting the names
fnames = []
for fname in os.listdir(data_dir):
    if fname.endswith('tiff')==False or fname[0]!='r':
        continue
    fnames.append(fname[15:-9])

# keeping only unique entries and sorting them
fnames = list(set(fnames))
fnames.sort()

# processing each timestep
f = open(out_file,'w')
for fname in fnames:
    print fname
    area = ProcessZip.run(fname,data_dir)
    if area==-1:
        continue
    datestr = fname[0:15]
    f.write(datestr + ',' + str(area) + '\n')

f.close()





