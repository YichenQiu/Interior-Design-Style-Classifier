from PIL import Image
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,accuracy_score

def fit_lgmodel(n_image=300,C=1.0):
    X,y=feature_label(n_image)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25)
    lgmodel=LogisticRegression(C=C,solver='sag',max_iter=10000)
    lgmodel.fit(X_train,y_train)
    pred_proba=lgmodel.predict_proba(X_test)
    pred=lgmodel.predict(X_test)
    log_loss_score=log_loss(y_test,pred_proba)
    accuracy=accuracy_score(y_test,pred)
    print ("log_loss score is {}".format(log_loss_score))
    print ("Accuracy score is {}".format(accuracy))
    return lgmodel

def feature_label(n_image):
    style_list=['Industrial','coastal','Bohemian','Scandinavian']
    X_list=[]
    y_arr=np.array([])
    for i,n in enumerate(style_list):
        X=feature_avgrgb(n,n_image,i)
        X_list+=X
        X_arr=np.array(X_list)
        y_arr=np.append(y_arr,np.zeros((n_image,))+i)
    return X_arr,y_arr


def feature_avgrgb(folder,n_image,label):
    '''Iterate through images in the folder
       Extract the RGB info from an image
       Calculate the mean R,G,B for each image
       Return a list of list of average R,G,B'''
    result_list=[]
    for image in os.listdir(folder)[:n_image]:
        im = Image.open("{}/{}".format(folder,image))
        im=im.resize((200,200))
        pixel_value=np.array(im.getdata())
        meanrbg=pixel_value.mean(axis=0)
        r,g,b=meanrbg[0],meanrbg[1],meanrbg[2]
        result_list.append([r,g,b])
    return result_list
