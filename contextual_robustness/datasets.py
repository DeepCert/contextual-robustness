import os, sys, pickle, typing
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from contextual_robustness.utils import normalize

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
GTSRB_PATH = os.path.join(DATA_PATH, 'gtsb')
OTHER_PATH = os.path.join(DATA_PATH, 'other')

def _fix_gtsrb_labels(labels:list) -> list:
    '''fixes the GTRSB labels after eliminating a subset of classes

    Args:
        labels (list): list

    Returns:
        list: list of fixed labels
    '''
    labels = labels -1
    for i in range(labels.shape[0]):
        if (labels[i] >5 ):
            labels[i] -= 1
    return labels

def load_gtsrb() -> typing.Tuple[np.array, np.array, np.array, np.array, pd.DataFrame]:
    '''Loads a subset of classes from the GTSRB dataset (classes = 1,2,3,4,5,7,8)

    Returns:
        tuple[np.array, np.array, np.array, np.array, pd.DataFrame]: (X_train, Y_train, X_test, Y_test, labels)
    '''
    train_path = os.path.join(GTSRB_PATH, 'train.p')
    test_path = os.path.join(GTSRB_PATH, 'test.p')
    labels_path = os.path.join(GTSRB_PATH, 'signnames.csv')

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['features'], train_data['labels']
    X_test,  y_test  = test_data['features'], test_data['labels']

    X_train, X_test = normalize(X_train), normalize(X_test)

    classes = [1,2,3,4,5,7,8]
    data = pd.read_csv(labels_path)
    labels = data[data['ClassId'].isin(classes)]

    mask = np.zeros_like(y_train)
    for c in classes:
        mask = np.logical_or(mask, y_train == c)

    X_train = X_train[mask]
    y_train = y_train[mask]

    y_train = _fix_gtsrb_labels(y_train)
    y_train = to_categorical(y_train)

    mask = np.zeros_like(y_test)
    for c in classes:
        mask = np.logical_or(mask, y_test == c)

    X_test = X_test[mask]
    y_test = y_test[mask]

    y_test = _fix_gtsrb_labels(y_test)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test, labels

def load_cifar() -> typing.Tuple[np.array, np.array, np.array, np.array, pd.DataFrame]:
    '''Loads the CIFAR dataset

    Returns:
        typing.Tuple[np.array, np.array, np.array, np.array, pd.DataFrame]: (X_train, Y_train, X_test, Y_test, labels)
    '''
    labels = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # normalize inputs between 0.0 and 1.0
    X_train, X_test = normalize(X_train.astype(np.float32)), normalize(X_test.astype(np.float32))
    # one-hot encode labels
    Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
    return X_train, Y_train, X_test, Y_test, labels

def load_nocex_image() -> np.array:
    '''Loads 'no counterexample' placeholder image

    Returns:
        np.array: placeholder image (width=256, height=256)
    '''
    image = None
    with open(os.path.join(OTHER_PATH, 'no-cex_256x256.p'), 'rb') as f:
        image = pickle.load(f)
    return image