import os
import random

def create_train_files(labels,photo,n):
    train_list=[]
    for label in labels:
        train_sample=random.sample(os.listdir("{}/{}".format(photo,label)),n)
        train_list.append(train_sample)

    for label_i,directory in enumerate(train_list):
        for image in directory:
            os.rename("{}/{}/{}".format(photo,labels[label_i],image), "train/{}/{}".format(labels[label_i],image))

def create_test_files(labels,photo,n):
    test_list=[]
    for label in labels:
        print (label)
        test_sample=random.sample(os.listdir("{}/{}".format(photo,label)),n)
        test_list.append(test_sample)

    for label_i,directory in enumerate(test_list):
        for image in directory:
            os.rename("{}/{}/{}".format(photo,labels[label_i],image), "test/{}/{}".format(labels[label_i],image))

def create_train_test(labels,photo,total_n,ratio):
    test_n=int(total_n*ratio)
    train_n=int(total_n-test_n)
    print (test_n, train_n)
    create_train_files(labels,photo,train_n)
    create_test_files(labels,photo,test_n)

if __name__=="__main__":
    labels=["Bohemian","Coastal","Industrial","Scandinavian"]
    test="test"
    train="train"
    photo="photo"
    create_train_test(labels,photo,450,0.2)
