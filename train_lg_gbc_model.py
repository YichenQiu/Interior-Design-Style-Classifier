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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

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
    y_train1 = np.zeros((num_image, 4))
    y_train1[np.arange(num_image), y_train] = 1

    train_generator.reset
    X_train=model.predict_generator(train_generator,verbose=1)
    print (X_train.shape,y_train1.shape)
    return X_train,y_train1,model

def train_last_layer(img_width, img_height,
                        train_data_dir,
                        num_image,
                        epochs = 50):
    X_train,y_train,model=feature_extraction_InV3(img_width, img_height,
                            train_data_dir,
                            num_image,
                            epochs)

    my_model = Sequential()
    my_model.add(BatchNormalization(input_shape=X_train.shape[1:]))
    my_model.add(Dense(1024, activation = "relu"))
    my_model.add(Dense(4, activation='softmax'))
    my_model.compile(optimizer="SGD", loss='categorical_crossentropy',metrics=['accuracy'])
    #early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    my_model.fit(X_train, y_train,epochs=18,batch_size=30,verbose=1)
    # my_model=LogisticRegression(multi_class="multinomial",solver="newton-cg",max_iter=4000)
    # my_model.fit(X_train,y_train)
    return my_model,model

def evaluate_model(test_data_dir,my_model,model):
    test_generator=ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_data_dir,
    target_size = (299, 299),
    batch_size = 15,
    class_mode = "categorical",
    shuffle=False)
    y_test=test_generator.classes
    y_images=test_generator.filenames
    test_generator.reset

    X_test=model.predict_generator(test_generator,verbose=1)

    y_pred=my_model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred.argmax(axis=1))
    loss=log_loss(y_test,y_pred)
    print ("Accuracy score: {}".format (accuracy))
    print ("Log_loss score: {}".format (loss))
    return y_test,y_pred,y_images




if __name__=="__main__":
    img_width=299
    img_height = 299
    train_data_dir = "data/train"
    test_data_dir="data/test"
    num_image=1440
    epochs = 10
    my_model, model=train_last_layer(test_data_dir,img_width, img_height,
                            train_data_dir,
                            num_image,epochs)
    y_test,y_pred,y_images=evaluate_model(test_data_dir,my_model,model)

    model.save('inV3_last_layer.h5')
