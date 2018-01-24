import pickle as pk
import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

class LgModel(object):
    def __init__(self):
        self.X=None

    def _feature_avgrgb(self,image):
        '''Takes an image
            Returns its average RGB as feature matrix '''
        result_list=[]
        #im = Image.open(image)
        im=im.resize((200,200))
        pixel_value=np.array(im.getdata())
        meanrbg=pixel_value.mean(axis=0)
        r,g,b=meanrbg[0],meanrbg[1],meanrbg[2]
        result_list.append([r,g,b])
        return np.array(result_list)

    def predict(self,image):
        '''Takes an image
           Return the predicted class for the image as string'''
        style_list=['Industrial','coastal','Bohemian','Scandinavian']
        X=self._feature_avgrgb(image)
        self.X=X
        with open("lgModel.pkl",'rb') as f:
            lgmodel = pk.load(f)
        predictions=lgmodel.predict_proba(X)
        return style_list[np.argmax(pred)]
