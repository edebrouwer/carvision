import numpy as np
import cv2
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



def histo_grad(angle, mag):
    length_vec=mag.shape[0]
    modul20=np.mod(angle,20)
    first_bin=np.mod(angle-modul20,180)/20
    second_bin=np.mod(angle-modul20+20,180)/20
    first_prop=(1-modul20/20)
    mat=np.zeros((9,length_vec))
    mat[first_bin.astype(int),range(0,length_vec)]=first_prop*mag
    mat[second_bin.astype(int),range(0,length_vec)]=mat[second_bin.astype(int),range(0,length_vec)]+(1-first_prop)*mag
    return(np.sum(mat,axis=1))

def hog_compute(path=None,image=None,resize_fact=None):
    #Computation of gradients
    hog_length=9

    if path:
        im=cv2.imread(path)
    elif image.any():
        im=image
    else:
        raise Exception('No Arguments Given')

    if (resize_fact!=None): #resize image if asked
        im = cv2.resize(im,None,fx=resize_fact, fy=resize_fact, interpolation = cv2.INTER_CUBIC)

    im=np.float32(im)/255.0

    im=im[np.mod(im.shape[0],8):,np.mod(im.shape[1],8):,:]


    gx= cv2.Sobel(im, cv2.CV_32F,1,0,ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

    #Gradient is the max of the 3 channels. Angle is the angle of the max gradient
    mag_3, angle_3 = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag=np.amax(mag_3,axis=2)
    angleindex=np.argmax(mag_3,axis=2)

    y,x=np.ogrid[0:im.shape[0],0:im.shape[1]]
    angle=angle_3[y,x,angleindex]

    #Compute HOGs
    Grid_size=(8,8) #Grid size for gradients
    nb_width=int(im.shape[1]/Grid_size[1])
    nb_height=int(im.shape[0]/Grid_size[0])

    hog_tot=np.zeros((nb_height,nb_width,hog_length))
    for i in range(0,nb_height):
        for j in range(0,nb_width):
            sub_angle=angle[i*Grid_size[0]:(i+1)*Grid_size[0],j*Grid_size[1]:(j+1)*Grid_size[1]]
            sub_mag=mag[i*Grid_size[0]:(i+1)*Grid_size[0],j*Grid_size[1]:(j+1)*Grid_size[1]]
            hog_tot[i,j,:]=histo_grad(sub_angle,sub_mag)

    #Normalization
    Norm_grid=(16,16) #Grid size for normalization
    stride = 8 #pixels forward at each step
    fact_ratio=2 #1D ratio of norm grids and hog grids
    nb_x=int((im.shape[1]-Norm_grid[1])/stride)+1
    nb_y=int((im.shape[0]-Norm_grid[0])/stride)+1

    hog_norm=np.zeros((nb_y,nb_x,hog_length*(fact_ratio**2)))
    for i in range(0,nb_y):
        for j in range(0,nb_x):
            hog_vec=hog_tot[i:(i+fact_ratio),j:(j+fact_ratio),:].flatten()
            if (np.linalg.norm(hog_vec)!=0):
                hog_norm[i,j,:]=hog_vec/np.linalg.norm(hog_vec)
            else:
                hog_norm[i,j,:]=hog_vec
    return(hog_norm)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    #Attention Window Size in WidthxHeight

    #If window is does not fit perfectly in the image, overlap is created at the edges.
    for y in xrange(0, (image.shape[0]-windowSize[1]+1), stepSize):
        lower_bound=(y + windowSize[1])
        if ((y + windowSize[1])>image.shape[0]):
            lower_bound = image.shape[0]
            y=lower_bound-windowSize[1]
        for x in xrange(0, (image.shape[1]-windowSize[0]+1), stepSize):
            # yield the current window
            right_bound=(x + windowSize[0])
            if ((x + windowSize[0])>image.shape[1]):
                right_bound=image.shape[1]
                x=right_bound-windowSize[0]
            yield (x, y, image[y:lower_bound, x: right_bound])

def pyramid(image,scale=1.5,lowest=32):
    #Returns a generator of a pyramid of images with specified scale.
    #scale is the scale factor between images for one dimension. So surface will be scaled by scale^2
    #lowest is the smallest width of height of a scaled image.
    (h,w,c)=image.shape
    im_scaled=image
    scaling=1
    while(min(h,w)>=(lowest)):
        yield(im_scaled,scaling)
        im_scaled=cv2.resize(im_scaled,(0,0),fx=(1/scale),fy=(1/scale))
        scaling=scaling*scale
        (h,w,c)=im_scaled.shape

def hog_to_csv(output_file=None,HOG_dim=None,train_dir_pos=None,train_dir_neg=None,resize_fact=None):

    file_list_pos=os.listdir(train_dir_pos)
    file_list_neg=os.listdir(train_dir_neg)

    Xtrain=np.zeros((len(file_list_pos)+len(file_list_neg),HOG_dim))
    Ytrain=np.zeros((len(file_list_pos)+len(file_list_neg),1))
    idx=0
    for image in file_list_pos:
        Xtrain[idx,:]=hog_compute(path=(train_dir_pos+image),resize_fact=resize_fact).flatten()
        Ytrain[idx]=1
        idx+=1

    for image in file_list_neg:
        Xtrain[idx,:]=hog_compute(path=train_dir_neg+image,resize_fact=resize_fact).flatten()
        idx+=1

    df_lab=pd.DataFrame({"label":Ytrain.ravel()})
    df_hogs=pd.DataFrame(Xtrain)
    df_sample=pd.concat([df_lab,df_hogs],axis=1)

    #Write the csv file for the data-samples
    df_sample.to_csv(output_file)

def best_svm(csv_file,C,gamma):
    df=pd.DataFrame.from_csv(csv_file)
    Y=df.iloc[:,:1].as_matrix()
    X=df.iloc[:,1:].as_matrix()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=0.20, random_state=42)

    #SVM Classifier
    folds=5
    param_grid = {'C': C,'gamma': gamma }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid,cv=folds)
    clf.fit(X_train,Y_train)
    #Accuracy on Test Set:
    pred_test=clf.best_estimator_.predict(X_test)
    accuracy=abs(pred_test-Y_test.T).sum()/len(pred_test)
    print("TEST ACCURACY :   ")
    print(1-accuracy)
    return clf

def get_hogDim(width,height,resize_fact):
    new_width=int(width*resize_fact/8)*8
    new_height=int(height*resize_fact/8)*8
    return (new_width/8-1)*(new_height/8-1)*36


def main():
    from sklearn.externals import joblib

    print("Computing HOGS from pics folder")
    hog_to_csv(output_file="HOG_car.csv",HOG_dim=get_hogDim(64,64,0.5),train_dir_pos=".Pics/car/",train_dir_neg="./Pics/no_car",resize_fact=0.5)
    print("Hogs saved to csv")

    C=[1e3, 5e3, 1e4, 5e4, 1e5]
    gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

    print("Computing best svm classifier with grid :")
    print("C")
    print(C)
    print("Gamma")
    print(gamma)

    mod=best_svm("HOG_dam.csv",C,gamma)

    print("Saving Classifier")
    joblib.dump(mod,"GridCLF.pkl")
    joblib.dump(mod.best_estimator_,"BestCLF.pkl")

if __name__ == "__main__":main()
