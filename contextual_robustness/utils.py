import os, types
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax

def create_output_path(outpath:str):
    '''Creates any non-existent folder(s) in the outpath

    Args:
        outpath (str): Path to a file or directory
    '''
    dirpath, _ = os.path.split(outpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

def get_file_extension(filepath:str) -> str:
    '''Gets the extension from a filepath.

    Args:
        filepath (str): Path to the file

    Returns:
        str: The file's extension (e.g. '.txt')
    '''
    return Path(filepath).suffix

def remove_softmax_activation(model_path:str, save_path:str='') -> tf.keras.Model:
    '''Prepares a classifier with softmax activation for verification by 
    removing the softmax activation function from the output layer.

    Args:
        model_path (str): Path to model
        save_path (str, optional): Path where new model is saved. Defaults to ''.

    Returns:
        tf.keras.Model: The modified tensorflow Model object
    '''
    model = tf.keras.models.load_model(model_path)
    weights = model.get_weights()
    model.pop()
    model.add(tf.keras.layers.Dense(weights[-1].shape[0], name='dense_output'))
    model.set_weights(weights)
    if save_path:
        tf.saved_model.save(model, save_path)
    return model

def softargmax(y:np.array) -> np.array:
    '''Applies softmax & argmax to emulate the softmax output layer of a tensorflow model

    Args:
        y (np.array): Logits layer output

    Returns:
        np.array: onehot encoded prediction (e.g. [0,0,1,0])
    '''
    out = np.zeros(y.shape[0], dtype=int)
    out[np.argmax(softmax(y))] = 1
    return out

def parse_indexes(indexes_list:list=[]) -> list:
    '''Parses mixed list of integers and ranges from CLI into a discret list of integers.

    Args:
        indexes_list (list, optional): List of strings of mixed ints and/or ranges (e.g. ['1', '2', '5-7', '10-15']). Defaults to [].

    Returns:
        list: Discret list of unique integers
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

def set_df_dtypes(df:pd.DataFrame, dtypes:dict) -> pd.DataFrame:
    '''Sets datatypes for specified columns of DataFrame

    Args:
        df (pd.DataFrame): The DataFrame
        dtypes (dict): Dictionary of datatypes (e.g. {'col':type, ...})

    Returns:
        pd.DataFrame: The updated DataFrame
    '''
    for k,v in dtypes.items():
        df[k] = df[k].astype(v)
    return df
