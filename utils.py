import os, json
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

def remove_softmax_activation(model_path, save_path=''):
    '''
    prepares a classifier network with softmax activation for verification by
    removing the softmax activation function from the output layer.

    Arguments:
        model_path (string) - path to the original tensorflow model
        save_path  (string) - path where new model is saved
    
    Returns:
        tensorflow model
    '''
    model = tf.keras.models.load_model(model_path)
    weights = model.get_weights()
    model.pop()
    model.add(tf.keras.layers.Dense(weights[-1].shape[0], name='dense_output'))
    model.set_weights(weights)
    if save_path:
        tf.saved_model.save(model, save_path)
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
    out[np.argmax(softmax(y))] = 1
    return out

def parse_indexes(indexes_list=[]):
    '''
    parses list of mixed integers and ranges of integers and returns a list of unique integers

    Arguments:
        indexes_list ([str]) - list of strings of integers or ranges (e.g. 1, 2, 5-7, 10-15)
    
    Returns:
        list of integers
    '''
    indexes = []
    for item in indexes_list:
        pieces = item.split('-')
        assert len(pieces) >= 1, 'each index must be an integer or range (e.g. 1 or 1-5)'
        assert len(pieces) <= 2, 'range of integers must be in format START-END (e.g. 1-10)'
        if len(pieces) == 1:
            start = end = pieces[0]
        elif len(pieces) == 2:
            start, end = pieces
        start, end = int(start), int(end)
        indexes += list(range(start, end + 1))
    return set(indexes)

def set_df_dtypes(df, dtypes):
    for k,v in dtypes.items():
        df[k] = df[k].astype(v)
    return df
