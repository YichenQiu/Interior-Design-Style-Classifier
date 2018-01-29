import pickle as pk
import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import load_model
import cv2

import numpy as np

# class LgModel(object):
#     def __init__(self):
#         self.X=None
#
#     def _feature_avgrgb(self,image):
#         '''Takes an image
#             Returns its average RGB as feature matrix '''
#         result_list=[]
#         im = Image.open(image)
#         im=im.resize((200,200))
#         pixel_value=np.array(im.getdata())
#         meanrbg=pixel_value.mean(axis=0)
#         r,g,b=meanrbg[0],meanrbg[1],meanrbg[2]
#         result_list.append([r,g,b])
#         return np.array(result_list)
#
#     def predict(self,image):
#         '''Takes an image
#            Return the predicted class for the image as string'''
#         style_list=['Industrial','coastal','Bohemian','Scandinavian']
#         X=self._feature_avgrgb(image)
#         self.X=X
#         with open("lgModel.pkl",'rb') as f:
#             lgmodel = pk.load(f)
#         pred=lgmodel.predict_proba(X)
#         return style_list[np.argmax(pred)]

class inception_retrain(object):
    def __init__(self):
        self.img=None
        self.model=None
        self.InV3model=None
    def _load_image(self,img):
        '''Takes an image
            Returns its proper form to feed into model's predcition '''
        #image = cv2.imread('test/{}'.format(img))
        nparr = np.fromstring(img, np.uint8)
        image = cv2.imdecode(nparr, -1)
        image = cv2.resize(image, (299, 299))
        image = np.expand_dims(image/255, axis=0)
        image = np.vstack([image])
        return image

    def _feature_extraction_inception(self,img):
        image=self._load_image(img)
        self.img=image
        #model = load_model('inception.h5')
        # model.compile(loss='categorical_crossentropy,
        #       optimizer='rmsprop',
        #       metrics=['accuracy'])
        features=self.InV3model.predict(image)
        return features

    def _load_model(self):
        if self.model is None:
            self.model=load_model('inV3_last_layer.h5')
        if self.InV3model is None:
            self.InV3model=load_model("inception.h5")

    def predict(self,img):
        '''Takes an imagebbb
           Return the predicted probabilities for each class'''
        self._load_model()
        image=self._feature_extraction_inception(img)
        self.img=image
        #self._load_model()
        # model.compile(loss='categorical_crossentropy,
        #       optimizer='rmsprop',
        #       metrics=['accuracy'])
        pred=self.model.predict(image)
        pred=np.round(pred,3).reshape(4,)
        return pred[0],pred[1],pred[2],pred[3]
