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
    batch_size = 20,
    class_mode = "categorical",
    shuffle=False)

    y_train=train_generator.classes
    y_train1 = np.zeros((num_image, 4))
    y_train1[np.arange(num_image), y_train] = 1

    train_generator.reset
    X_train=model.predict_generator(train_generator,verbose=1)
    return X_train,y_train1

def train_last_layer(img_width, img_height,
                        train_data_dir,
                        num_image,
                        epochs = 50):
    X_train,y_train=feature_extraction_InV3(img_width, img_height,
                            train_data_dir,
                            num_image,
                            epochs)
    my_model = Sequential()
    my_model.add(BatchNormalization(input_shape=X_train.shape[1:]))
    #my_model.add(Dense(1024, activation = "relu"))
    my_model.add(Dense(4, activation='softmax',kernel_regularizer=regularizers.l2(10)))
    sgd=optimizers.SGD(lr=0.0001)
    my_model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    my_model.fit(X_train, y_train,epochs=600,batch_size=30,validation_split=0.15,verbose=1,callbacks=[early])
    return my_model

if __name__=="__main__":
    img_width=299
    img_height = 299
    train_data_dir = "data/train"
    num_image=1439
    epochs = 50
    model=train_last_layer(img_width, img_height,
                            train_data_dir,
                            num_image,epochs)

    model.save('inV3_last_layer.h5')
    #with open("InV3LastLayerModel.pkl", 'wb') as f:
        #pk.dump(model, f)
