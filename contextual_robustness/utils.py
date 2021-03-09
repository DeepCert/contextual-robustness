import os, typing, time, cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax

def set_tf_log_level(level:int=1):
    log_levels = {0: 'FATAL', 1: 'ERROR', 2: 'WARN', 3: 'INFO', 4: 'DEBUG'}
    assert level in log_levels.keys(), f'unsupported TF log level. supported:{log_levels.keys()}'
    tf.get_logger().setLevel(log_levels.get(level))

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

def normalize(X:np.array) -> np.array:
    '''normalizes image values between 0.0 and 1.0

    Args:
        X (np.array): array of images

    Returns:
        np.array: normalized images
    '''
    return X / 255.0

def resize_image(image:np.array, size:typing.Tuple[int, int]) -> np.array:
    '''resizes an image

    Args:
        image (np.array): the original image
        size (typing.Tuple[int, int]): size of resized image (width, height)

    Returns:
        np.array: resized image
    '''
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)

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

def parse_indexes(indexes_list:typing.List[str], sort=True) -> typing.List[int]:
    '''Parses mixed list of integers and ranges from CLI into a discret list of integers.

    Args:
        indexes_list (list, optional): List of strings of mixed ints and/or ranges (e.g. ['1', '2', '5-7', '10-15']). Defaults to [].

    Returns:
        list[int]: Discret list of unique integers
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
    # remove any duplicates and sort if necessary
    return sorted(list(set(indexes))) if sort else list(set(indexes))

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

def ms_to_human(ms:int) -> str:
    '''converts milliseconds to human-readable string

    Args:
        ms (int): number of milliseconds

    Returns:
        str: human-readable string in format "[h hours], [m minutes], [s seconds]" OR "[ms milliseconds]"
    '''
    if ms < 1000:
        return f'{ms} milliseconds'
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    hours = int((ms / (1000 * 60 * 60)) % 24)
    output = f'{seconds} seconds'
    output = f'{minutes} minutes, {output}' if minutes or hours else output
    output = f'{hours} hours, {output}' if hours else output
    return output

class Timer:
    '''A simple timer class'''
    def __init__(self, autostart:bool=False) -> object:
        '''Constructor for Timer class

        Args:
            autostart (bool): auto-start the timer

        Returns:
            object: Timer object
        '''
        self._start_time, self._end_time = 0, 0
        if autostart: self.start()

    @property
    def start_time(self) -> int:
        '''start_time property

        Returns:
            int: the start_time (ms since 1970)
        '''
        return self._start_time

    @property
    def end_time(self) -> int:
        '''end_time property

        Returns:
            int: the end_time (ms since 1970)
        '''
        return self._end_time

    def get_elapsed(self, as_string:bool=False) -> int:
        '''elapsed time property (as string using get_elapsed(as_string=True))

        Args:
            as_string (bool, optional): returns as string with unit (ms). Defaults to False.

        Returns:
            int: milliseconds between start_time and end_time
        '''
        elapsed = round(self.end_time - self.start_time)
        if as_string:
            return f'{elapsed}ms'
        return elapsed
    elapsed = property(get_elapsed)

    @start_time.setter
    def start_time(self, t:int):
        '''start_time property setter

        Args:
            t (int): start_time (ms since 1970)
        '''
        self._start_time = t

    @end_time.setter
    def end_time(self, t:int):
        '''end_time property setter

        Args:
            t (int): end_time (ms since 1970)
        '''
        self._end_time = t

    def _timestamp_ms(self) -> int:
        '''gets current timestamp (ms since 1970)

        Returns:
            int: current timestamp (ms since 1970)
        '''
        return time.time() * 1000

    def start(self) -> object:
        '''start the timer

        Returns:
            object: the Timer object
        '''
        self.start_time = self._timestamp_ms()

    def end(self) -> int:
        '''stops the timer

        Returns:
            int: elapsed time (ms)
        '''
        self.end_time = self._timestamp_ms()
        return self.elapsed
