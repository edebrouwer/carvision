

import cv2
import os
import csv

#dir_csv is path to csv with names of the files
#if mode is "all" then label all cars if mode is blank : label blanks.
def label_data(dir_csv,dir_pics,mode):

    #List of file names
    names_list=[]
    with open(dir_csv, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            names_list.append(row)
    if (mode=="blank"):
        index=0
        print("PRESS Q TO QUIT")
        for f in names_list:
            if(f[1]=="B"):
                im=cv2.imread(dir_pics+f[0])
                cv2.imshow('image',im)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
                var=raw_input("Press D for damaged or C for clean :      ")
                if(var=="Q"):
                    break
                f[1]=var.upper()
    with open(dir_csv, "wb") as f:
        writer = csv.writer(f,dialect='excel',delimiter=',')
        for item in names_list:
            print(item)
            writer.writerow(item)


def main():
    label_data(dir_csv="./pics_S.csv",dir_pics="~/ENV/pics_S/",mode="blank")

if __name__ == "__main__":
    main()
