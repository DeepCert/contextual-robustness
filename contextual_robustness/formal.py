import os, sys, typing
import numpy as np
import pandas as pd
import tensorflow as tf
from contextual_robustness.utils import softargmax, _get_file_extension
from contextual_robustness.base import _BaseContextualRobustness, Techniques, ContextualRobustness, DEFAULTS

# TODO: remove sys.path.append after maraboupy pip package is available.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../marabou')))
from maraboupy import Marabou

# ======================================================================
# ContextualRobustnessFormal
# ======================================================================
class ContextualRobustnessFormal(_BaseContextualRobustness):
    '''Class for ContextualRobustness 'Formal Verification' analysis

    Args:
        model_path (str, optional): Path to saved tensorflow model. Defaults to ''.
        model_name (str, optional): Name of model. Defaults to ''.
        model_args (dict, optional): Args passed to Marabou to load network. Defaults to dict().
        X (np.array, optional): The images. Defaults to np.array([]).
        Y (np.array, optional): Labels for images (onehot encoded). Defaults to np.array([]).
        sample_indexes (list[int], optional): List of indexes to test from X. Defaults to [].
        transform_fn (callable, optional): The image transform function (required args: x, epsilon). Defaults to lambda x:x.
        transform_args (dict, optional): Additional arguments passed to transform_fn. Defaults to dict().
        transform_name (str, optional): Name of transform. Defaults to ''.
        eps_lower (float, optional): Min possible epsilon. Defaults to 0.0.
        eps_upper (float, optional): Max possible epsilon. Defaults to 1.0.
        eps_interval (float, optional): Step size between possible epsilons. Defaults to 0.002.
        marabou_options (dict, optional): Additional MarabouOptions. Defaults to dict(verbosity=0).
        verbosity (int, optional): Amount of logging (0-4). Defaults to 0.

    Returns:
        ContextualRobustness: the ContextualRobustnessFormal object
    '''
    def __init__(
        self,
        model_path:str='',
        model_name:str='',
        model_args:dict=dict(),
        transform_fn:callable=lambda x: x,
        transform_args:dict=dict(),
        transform_name:str='',
        X:np.array=np.array([]),
        Y:np.array=np.array([]),
        sample_indexes:list=[],
        eps_lower:float=DEFAULTS['eps_lower'],
        eps_upper:float=DEFAULTS['eps_upper'],
        eps_interval:float=DEFAULTS['eps_interval'],
        marabou_options:dict=dict(verbosity=DEFAULTS['marabou_verbosity']),
        verbosity:int=DEFAULTS['verbosity']
        ) -> ContextualRobustness:
        self._model_args = model_args
        self._marabou_options = marabou_options
        # Execute the superclass's constructor
        super().__init__(
            model_path=model_path,
            model_name=model_name,
            transform_fn=transform_fn,
            transform_args=transform_args,
            transform_name=transform_name,
            X=X,
            Y=Y,
            sample_indexes=sample_indexes,
            eps_lower=eps_lower,
            eps_upper=eps_upper,
            eps_interval=eps_interval,
            verbosity=verbosity
            )
    
    @property
    def technique(self) -> Techniques:
        '''technique property

        Returns:
            Techniques: verification technique (Techniques.FORMAL)
        '''
        return Techniques.FORMAL
    
    def _load_model(self, model_path:str) -> Marabou.MarabouNetwork:
        '''Loads a tensorflow, nnet, or onnx model as a MarabouNetwork

        Args:
            model_path (str): Path to the verification model

        Returns:
            MarabouNetwork: the MarabouNetwork object
        '''
        valid_exts = ('.nnet', '', '.pb', '.h5', '.hdf5', '.onnx')
        ext = _get_file_extension(model_path)
        assert ext in valid_exts, 'Model must be .nnet, .pb, .h5, or .onnx'
        if ext == '.nnet':
            return Marabou.read_nnet(model_path, **self._model_args)
        elif ext in ('', '.pb', '.h5', '.hdf5'):
            return Marabou.read_tf(model_path, **self._model_args)
        elif ext == '.onnx':
            return Marabou.read_onnx(model_path, **self._model_args)
        return None
    
    def _find_correct_sample_indexes(self, X:np.array, Y:np.array) -> typing.List[int]:
        '''Finds list of indexes for correctly predicted samples

        Args:
            X (np.array): The images
            Y (np.array): Labels (onehot encoded) for the images

        Returns:
            list[int]: Indexes of correctly predicted samples from dataset
        '''
        ext = _get_file_extension(self._model_path)
        if ext == '.nnet' or ext == '.onnx':
            # For NNet & ONNX models, use Marabou's 'evaluate' to make predictions
            return [i for i in self._sample_indexes if np.argmax(softargmax(self._model.evaluate(X[i])[0])) == np.argmax(Y[i])]
        elif ext in ('', '.pb', '.h5', '.hdf5'):
            # For Tensorflow models, use Tensorflow's 'predict' to make predictions (much faster)
            model = tf.keras.models.load_model(self._model_path)
            Y_p = model.predict(np.array([X[si] for si in self._sample_indexes]))
            return [si for i,si in enumerate(self._sample_indexes) if np.argmax(softargmax(Y_p[i])) == np.argmax(Y[si])]
        return None
    
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
        counterexample = None
        while ((upper - lower) > interval):
            guess = lower + (upper - lower) / 2.0
            verified, pred, cex = self._find_counterexample(x, y, guess, x_index=index)
            if self._verbosity > 1:
                print(f'evaluated image:{index}@epsilon:{guess}, label:{actual_label}, pred:{pred}')
            if verified:
                # correct prediction
                lower = guess
            else:
                # incorrect prediction
                upper = guess
                predicted_label = pred
                epsilon = guess
                counterexample = cex
        return lower, upper, epsilon, predicted_label, counterexample
    
    def _find_counterexample(self, x:np.array, y:np.array, epsilon:float, x_index:int=None) -> typing.Tuple[bool, int, np.array]:
        '''Finds the the counterexample for an image at a given epsilon

        Args:
            x (np.array): The image
            y (np.array): Label for the image (onehot encoded)
            epsilon (float): the epsilon value
            x_index (int, optional): Index of x (for reference only). Defaults to None.

        Returns:
            tuple[bool, int, np.array]: (verified, predicted_label, counterexample)
        '''
        actual_label = np.argmax(y)
        predicted_label = actual_label
        verified = True
        counterexample = None
        for output_index in range(y.shape[0]):
            if actual_label == output_index:
                continue
            # load model, encode the transform as a marabou input query, and solve query
            network = self._load_model(self._model_path)
            network = self._transform_fn(network, x, epsilon, output_index, **self._transform_args)
            vals, stats = network.solve(options=Marabou.createOptions({'verbosity': 0, **self._marabou_options}), verbose=(self._verbosity > 3))
            # check results
            if stats.hasTimedOut():
                # TIMEOUT
                verified = False
                assert False, f'Timeout occurred ({f"x_index={x_index}" if x_index is not None else ""};output={output_index}@epsilon={epsilon})'
            elif len(vals) == 0:
                # UNSAT
                if self._verbosity > 2:
                    print(f'image:{x_index};output:{output_index}@epsilon:{epsilon} (UNSAT)')
                continue
            else:
                # SAT (counterexample found)
                if self._verbosity > 2:
                    print(f'image:{x_index};output:{output_index}@epsilon:{epsilon} (SAT)')
                counterexample = np.array([vals[i] for i in range(self.n_pixels)]).reshape(self.image_shape)
                predicted_label = output_index
                verified = False
                break
        return verified, predicted_label, counterexample
