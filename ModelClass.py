import pickle as pk
import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import load_model
from keras.preprocessing import image

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

    def load_image(self,img):
        '''Takes an image
            Returns its proper form to feed into model's predcition '''
        im=image.load_img('test/{}'.format(img), target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x/255, axis=0)
        image = np.vstack([x])
        return image

    def _feature_extraction_inception(self,img):
        image=self._load_image(img)
        self.img=image
        model = load_model('inception.h5')
        # model.compile(loss='categorical_crossentropy,
        #       optimizer='rmsprop',
        #       metrics=['accuracy'])
        features=model.predict(image)
        return features

    def predict(self,img):
        '''Takes an image
           Return the predicted probabilities for each class'''
        image=self._feature_extraction_inception(img)
        self.image=image
        model = load_model('inception_model.h5')
        # model.compile(loss='categorical_crossentropy,
        #       optimizer='rmsprop',
        #       metrics=['accuracy'])
        pred=model.predict(image)
        return pred
