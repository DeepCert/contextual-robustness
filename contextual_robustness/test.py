import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from contextual_robustness.base import _BaseContextualRobustness, ContextualRobustness, Techniques, defaults

# ======================================================================
# ContextualRobustnessTest
# ======================================================================
class ContextualRobustnessTest(_BaseContextualRobustness):
    '''Class for ContextualRobustness 'Test Based' analysis

    Args:
        model_path (str, optional): Path to saved tensorflow model. Defaults to ''.
        model_name (str, optional): Name of model. Defaults to ''.
        X (np.array, optional): The images. Defaults to np.array([]).
        Y (np.array, optional): Labels for images (onehot encoded). Defaults to np.array([]).
        sample_indexes (list[int], optional): List of indexes to test from X. Defaults to [].
        transform_fn (callable, optional): The image transform function (required args: x, epsilon). Defaults to lambda x:x.
        transform_args (dict, optional): Additional arguments passed to transform_fn. Defaults to dict().
        transform_name (str, optional): Name of transform. Defaults to ''.
        eps_lower (float, optional): Min possible epsilon. Defaults to 0.0.
        eps_upper (float, optional): Max possible epsilon. Defaults to 1.0.
        eps_interval (float, optional): Step size between possible epsilons. Defaults to 0.002.
        verbosity (int, optional): Amount of logging (0-4). Defaults to 0.

    Returns:
        ContextualRobustness: the ContextualRobustnessTest object
    '''
    def __init__(
        self,
        model_path:str= '',
        model_name:str='',
        X:np.array=np.array([]),
        Y:np.array=np.array([]),
        sample_indexes:list=[],
        transform_fn:callable=lambda x: x,
        transform_args:dict=dict(),
        transform_name:str='',
        eps_lower:float=defaults['eps_lower'],
        eps_upper:float=defaults['eps_upper'],
        eps_interval:float=defaults['eps_interval'],
        verbosity:int=defaults['verbosity']
        ) -> ContextualRobustness:
        # Execute the superclass's constructor
        super().__init__(
            model_path=model_path,
            model_name=model_name,
            X=X,
            Y=Y,
            sample_indexes=sample_indexes,
            transform_fn=transform_fn,
            transform_args=transform_args,
            transform_name=transform_name,
            eps_lower=eps_lower,
            eps_upper=eps_upper,
            eps_interval=eps_interval,
            verbosity=verbosity)
    
    @property
    def technique(self) -> Techniques:
        '''technique property

        Returns:
            Techniques: verification technique (Techniques.TEST)
        '''
        return Techniques.TEST
    
    def _load_model(self, model_path:str) -> tf.keras.Model:
        '''Loads a tensorflow (keras) model

        Args:
            model_path (str): Path to the saved model

        Returns:
            tf.keras.Model: tensorflow Model object
        '''
        return tf.keras.models.load_model(model_path)
    
    def _find_correct_sample_indexes(self, X:np.array, Y:np.array) -> typing.List[int]:
        '''Finds list of indexes for correctly predicted samples

        Args:
            X (np.array): The images
            Y (np.array): Labels (onehot encoded) for the images

        Returns:
            list[int]: Indexes of correctly predicted samples from dataset
        '''
        Y_p = self._model.predict(np.array([X[si] for si in self._sample_indexes]))
        return [si for i,si in enumerate(self._sample_indexes) if np.argmax(Y_p[i]) == np.argmax(Y[si])]
    
    def _find_epsilon(self, x:np.array, y:np.array, index:int=None) -> typing.Tuple[float, float, float, int, np.array]:
        '''Finds the epsilon for an image

        Args:
            x (np.array): The image
            y (np.array): Label for the image (onehot encoded)
            index (int, optional): Index of x. Defaults to None.

        Returns:
            tuple[float, float, float, int, np.array]: (lower, upper, epsilon, predicted_label, counterexample)
        '''
        lower = self._eps_lower
        upper = self._eps_upper
        interval = self._eps_interval
        actual_label = np.argmax(y)
        predicted_label = actual_label
        epsilon = upper
        counterexample = self.transform_image(x, epsilon)
        while ((upper - lower) > interval):
            guess = lower + (upper - lower) / 2.0
            x_trans = self.transform_image(x, guess)
            pred = np.argmax(self._model.predict(x_trans.reshape((1,) + x_trans.shape)))
            if self._verbosity > 1:
                print(f'evaluating image:{index}@epsilon:{guess}, label:{actual_label}, pred:{pred}')
            if pred == actual_label:
                # correct prediction
                lower = guess
            else:
                # incorrect prediction
                upper = guess
                predicted_label = pred
                epsilon = guess
                counterexample = x_trans
        return lower, upper, epsilon, predicted_label, counterexample
    
    def transform_image(self, x:np.array, epsilon:float) -> np.array:
        '''Transforms an image using transform_fn

        Args:
            x (np.array): The image
            epsilon (float): amount of transform

        Returns:
            np.array: The transformed image
        '''        
        return self._transform_fn(x, epsilon=epsilon, **self._transform_args)
