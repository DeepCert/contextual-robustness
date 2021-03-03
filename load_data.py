import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

def normalize(X):
    return X / 255.0

def fix_labels(labels):
    labels = labels -1
    for i in range(labels.shape[0]):
        if (labels[i] >5 ):
            labels[i] -= 1
    return labels

def loadTraffic():
    with open('data/gtsb/train.p', 'rb') as f:
        train_data = pickle.load(f)

    with open('data/gtsb/test.p', 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['features'], train_data['labels']
    X_test,  y_test  = test_data['features'], test_data['labels']

    data = pd.read_csv('data/gtsb/signnames.csv')

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    classes = [1,2,3,4,5,7,8]
    labels = data[data['ClassId'].isin(classes)]

    mask = np.zeros_like(y_train)
    for i in classes:
        mask = np.logical_or(mask, y_train == i)

    X_train = X_train[mask]
    y_train = y_train[mask]

    y_train = fix_labels(y_train)
    y_train = to_categorical(y_train)


    mask = np.zeros_like(y_test)
    for i in classes:
        mask = np.logical_or(mask, y_test == i)

    X_test = X_test[mask]
    y_test = y_test[mask]

    y_test = fix_labels(y_test)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test, labels

def load_cifar():
    labels = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # normalize inputs between 0.0 and 1.0
    X_train, X_test = normalize(X_train.astype(np.float32)), normalize(X_test.astype(np.float32))
    # one-hot encode labels
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
    return X_train, Y_train, X_test, Y_test, labels
