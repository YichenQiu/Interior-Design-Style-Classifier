from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D,BatchNormalization
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
import numpy as np
import pickle as pk
from keras import regularizers
from sklearn.metrics import accuracy_score,log_loss

def feature_extraction_InV3(img_width, img_height,
                        train_data_dir,
                        num_image,
                        epochs):
    base_model = InceptionV3(input_shape=(299, 299, 3),
                              weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=base_model.input, outputs=x)


    train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_data_dir,
    target_size = (299, 299),
    batch_size = 15,
    class_mode = "categorical",
    shuffle=False)

    y_train=train_generator.classes
    #y_train1 = np.zeros((num_image, 4))
    #y_train1[np.arange(num_image), y_train] = 1

    train_generator.reset
    X_train=model.predict_generator(train_generator,verbose=1)
    print (X_train.shape,y_train.shape)
    return X_train,y_train,model

def train_last_layer(test_data_dir,img_width, img_height,
                        train_data_dir,
                        num_image,
                        epochs = 50):
    X_train,y_train,model=feature_extraction_InV3(img_width, img_height,
                            train_data_dir,
                            num_image,
                            epochs)
    my_model = Sequential()
    my_model.add(BatchNormalization(input_shape=X_train.shape[1:]))
    my_model.add(Dense(256, activation = "relu"))
    my_model.add(Dense(4, activation='softmax'))
    my_model.compile(optimizer="SGD", loss='categorical_crossentropy',metrics=['accuracy'])
    #early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    my_model.fit(X_train, y_train,epochs=100,batch_size=30,validation_split=0.1,verbose=1)
    test_generator=ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_data_dir,
    target_size = (299, 299),
    batch_size = 15,
    class_mode = "categorical",
    shuffle=False)
    y_test=test_generator.classes
    test_generator.reset
    X_test=model.predict_generator(test_generator,verbose=1)
    y_pred=my_model.predict(X_test)
    y_pred_prob=my_model.predict_proba(X_test)
    accuracy=accuracy_score(y_test,y_pred.argmax(axis=1))
    loss=log_loss(y_test,y_pred_prob)
    print ("Accuracy score: {}".format (accuracy))
    print ("Log_loss score: {}".format (loss))
    return my_model

if __name__=="__main__":
    img_width=299
    img_height = 299
    train_data_dir = "data/train"
    test_data_dir="data/test"
    num_image=1439
    epochs = 10
    model=train_last_layer(test_data_dir,img_width, img_height,
                            train_data_dir,
                            num_image,epochs)

    model.save('inV3_last_layer.h5')
    #with open("InV3LastLayerModel.pkl", 'wb') as f:
        #pk.dump(model, f)
