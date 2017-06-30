
#Initialize csvs with name of all files per directory +B for the label.

import csv
import os

#Directories to initialize
dir_stack = ["pics_S","pics_erepS","pics_carexportS"]
pic_path= "~/ENV/PIC/"

for dire in dir_stack:
    with open("./"+dire+".csv", "wb") as f:
        writer = csv.writer(f,dialect='excel')
        for item in os.listdir(pic_path+dire):
            writer.writerow([item, "B"])
