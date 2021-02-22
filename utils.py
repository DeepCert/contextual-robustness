import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from scipy.special import softmax

def create_output_path(outpath):
    '''
    creates any non-existent folder(s) in the outpath

    Arguments:
        outpath (string) - path to a file or directory
    '''
    dirpath, _ = os.path.split(outpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

def get_file_extension(filepath):
    '''
    returns the extension of a file.

    Arguments:
        filepath (string) - path to the file
    
    Returns:
        extension (string) - the file's extension (e.g. '.txt')
    '''
    return Path(filepath).suffix

def prepare_classifier(model_path):
    '''
    prepares a classifier network with softmax activation for verification by
    removing the softmax activation function from the last layer.

    Arguments:
        model_path (string) - path to the original tensorflow model
    
    Returns:
        tensorflow model
    '''
    model = tf.keras.models.load_model(model_path)
    weights = model.get_weights()
    model.pop()
    model.add(tf.keras.layers.Dense(weights[-1].shape[0], name='dense_output'))
    model.set_weights(weights)
    return model

def softargmax(y):
    '''
    applies softmax & argmax to emulate the softmax output layer of a tensorflow model

    Arguments:
        y (np.array) - logits layer output
    
    Returns:
        (np.array) - onehot encoded prediction (e.g. [0,0,1,0])
    '''
    out = np.zeros(y.shape[0], dtype=int)
    y = softmax(y)
    out[np.argmax(y)] = 1
    return out
