import sys, enum, copy, pickle, typing
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import set_df_dtypes, create_output_path, softargmax, get_file_extension
from abc import ABCMeta, abstractmethod

# TODO: remove sys.path.append after maraboupy pip package is available.
sys.path.append('./marabou')
from maraboupy import Marabou

class Techniques(enum.Enum):
    '''Verification techniques enum'''
    TEST='test'
    FORMAL='formal'

# Generic type to encompass different ContextualRobustness objects
ContextualRobustness = typing.TypeVar('ContextualRobustness')

# Default values
defaults = dict(
    eps_lower=0.0,
    eps_upper=1.0,
    eps_interval=0.002,
    verbosity=0,
    marabou_verbosity=0
    )

# Datatypes for results DataFrame columns
results_dtypes = {
    'image': np.int64,
    'class': np.int64,
    'predicted': np.int64,
    'epsilon': np.float64,
    'upper': np.float64,
    'lower': np.float64
    }

# ======================================================================
# _BaseContextualRobustness
# ======================================================================
class _BaseContextualRobustness(metaclass=ABCMeta):
    '''Contains common functionality, properties, and defines abstract methods to be implemented by subclasses.

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
        ContextualRobustness: the ContextualRobustness object
    '''
    def __init__(
        self,
        model_path='',
        model_name='',
        transform_fn=lambda x: x,
        transform_args=dict(),
        transform_name='',
        X=np.array([]),
        Y=np.array([]),
        sample_indexes=[],
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval'],
        verbosity=defaults['verbosity'],
        ) -> ContextualRobustness:
        assert bool(model_path), 'model_path is required'
        assert X.shape[0] == Y.shape[0], 'X and Y must have equal number of items'
        assert callable(transform_fn), 'transform_fn must be callable (e.g. a function)'
        
        self._verbosity = verbosity
        self._model_path = model_path
        self._model_name = model_name
        self._transform_fn = transform_fn
        self._transform_args = transform_args
        self._transform_name = transform_name
        self._X, self._Y = X, Y
        self._sample_indexes = sample_indexes if len(sample_indexes) > 0 else range(X.shape[0])
        self._eps_lower = eps_lower
        self._eps_upper = eps_upper
        self._eps_interval = eps_interval

        self._model = self._load_model(model_path)
        # find indexes of correctly predicted samples
        self._correct_sample_indexes = self._find_correct_sample_indexes(X, Y)
        print(f'filtered {len(self._sample_indexes) - len(self._correct_sample_indexes)} incorrectly predicted samples')
        # measure accuracy 
        self._accuracy = len(self._correct_sample_indexes) / len(self._sample_indexes)
        print(f'accuracy on {len(self._sample_indexes)} samples is {round(self.accuracy * 100, 2)}%')
        # examples of images @ epsilon where network's prediction changed
        self._counterexamples = dict()

    @property
    @abstractmethod
    def technique(self) -> Techniques:
        '''technique property

        Returns:
            Techniques: verification technique (e.g. Techniques.TEST or Techniques.FORMAL)
        '''
        return None

    @property
    def model_name(self) -> str:
        '''model_name property

        Returns:
            str: name of model
        '''        
        return self._model_name
    
    @property
    def transform_name(self) -> str:
        '''transform_name property

        Returns:
            str: name of transform
        '''        
        return self._transform_name
    
    @property
    def classes(self) -> typing.List[int]:
        '''classes property

        Returns:
            list: list of integers representing classes in dataset
        '''
        return sorted(np.unique([np.argmax(self._Y[i]) for i in range(self._Y.shape[0])]))
    
    @property
    def dataset(self) -> typing.Tuple[np.array, np.array]:
        '''dataset property

        Returns:
            tuple[np.array, np.array]: tuple containing X and Y
        '''
        return self._X, self._Y
    
    @property
    def image_shape(self) -> typing.Tuple[int]:
        '''image_shape property

        Returns:
            tuple: shape of images in X
        '''
        return self.dataset[0].shape[1:]
    
    @property
    def n_pixels(self) -> int:
        '''n_pixels property

        Returns:
            int: number of pixels in each image.
        '''
        prod = 1
        for dim in self.image_shape:
            prod *= dim
        return prod

    @property
    def counterexamples(self) -> typing.Dict[str, np.array]:
        '''counterexamples property

        Returns:
            dict[str:np.array]: counterexamples for each image (e.g. {'image1': np.array([...]), ...})
        '''        
        return self._counterexamples

    def get_counterexample(self, x_index:int) -> np.array:
        '''Gets counterexample for an image by index

        Args:
            x_index ([int]): index of image in X

        Returns:
            np.array: the counterexample (or 'None' if does not exist)
        '''        
        return self.counterexamples.get(f'image{x_index}')
    
    def save_counterexample(self, x_index:int, counterexample:np.array):
        '''Saves the counterexample for an image

        Args:
            x_index ([int]): index of image
            counterexample ([np.array]): the counterexample to save
        '''        
        self._counterexamples[f'image{x_index}'] = counterexample
    
    def get_num_samples(self, class_index:int=None) -> int:
        '''Gets the number of samples under analysis (optionally for a single class using "get_num_samples").

        Args:
            class_index (int, optional): Index of class to get samples for. Defaults to None.

        Returns:
            int: Number of samples (for a particular class if class_index is supplied)
        '''
        if class_index is not None:
            return len([si for si in self._sample_indexes if np.argmax(self.dataset[1][si]) == class_index])
        return len(self._sample_indexes)
    num_samples = property(get_num_samples)

    def get_accuracy(self, class_index:int=None) -> float:
        '''Gets the accuracy (optionally for a single class using "get_accuracy")

        Args:
            class_index (int, optional): Index of class to get accuracy for. Defaults to None.

        Returns:
            float: Accuracy of model on the samples (for a particular class if class_index is supplied)
        '''
        if class_index is not None:
            sample_indexes = [si for si in self._sample_indexes if np.argmax(self._Y[si]) == class_index]
            correct_sample_indexes = [i for i in self._correct_sample_indexes if np.argmax(self._Y[i]) == class_index]
            return len(correct_sample_indexes) / len(sample_indexes) if len(sample_indexes) > 0 else 0
        return self._accuracy
    accuracy = property(get_accuracy)

    def get_results(self, class_index:int=None, sort_by:list=[]) -> pd.DataFrame:
        '''Gets result data (optionally for a single class, and optionally sorted by column using "get_results")

        Args:
            class_index (int, optional): Index of class to get results for. Defaults to None.
            sort_by (list[str], optional): Sorts results by one or more columns. Defaults to [].

        Returns:
            pd.DataFrame: The results (for a single class if class_index is supplied, and sorted by sort_by)
        '''
        results = self._results
        if class_index is not None:
            results = results[results['class'] == class_index]
        if len(sort_by) > 0:
            results = results.sort_values(by=sort_by)
        return results
    results = property(get_results)

    @abstractmethod
    def _find_epsilon(self, x:np.array, y:np.array, index:int=None) -> typing.Tuple[float, float, float, int, np.array]:
        '''Finds epsilon for a given image; Abstract method implemented by subclasses

        Args:
            x ([np.array]): the image
            y ([np.array]): categorical (onehot) encoded label for x
            index ([int], optional): Index of x in X (used for reference only). Defaults to None.

        Returns:
            tuple[float, float, float, int, np.array]: Tuple containing (lower, upper, epsilon, predicted_label, counterexample)
        '''
        pass

    @abstractmethod
    def _find_correct_sample_indexes(self, X:np.array, Y:np.array) -> typing.List[int]:
        '''Finds list of indexes for correctly predicted samples

        Args:
            X (np.array): The images
            Y (np.array): Labels (onehot encoded) for the images

        Returns:
            list[int]: Indexes of correctly predicted samples from dataset
        '''
        pass
    
    @abstractmethod
    def _load_model(self, model_path:str) -> object:
        '''Loads a model; Abstract method implemented by subclasses.

        Args:
            model_path (str): Path to model
        
        Returns:
            object: The model (either tf.keras.Model or MarabouNetwork)
        '''
        pass
    
    def analyze(self, epsilons_outpath:str='./epsilons.csv', counterexamples_outpath:str='./counterexamples.p') -> ContextualRobustness:
        '''Run analysis on the model & transform; Generates results csv and counterexamples pickle.

        Args:
            epsilons_outpath (str, optional): Path to csv file containing results. Defaults to './epsilons.csv'.
            counterexamples_outpath (str, optional): Path to pickle containing counterexamples. Defaults to './counterexamples.p'.

        Returns:
            ContextualRobustness: the object (self)
        '''
        print(f'analyzing {self.transform_name} on {len(self._correct_sample_indexes)} samples. this may take some time...')
        data = []
        for i in self._correct_sample_indexes:
            x, y = self._X[i], self._Y[i]
            actual_label = np.argmax(y)
            lower, upper, epsilon, predicted_label, counterexample = self._find_epsilon(x, y, index=i)
            data.append({
                'image': i,
                'class': actual_label,
                'predicted': predicted_label,
                'epsilon': epsilon,
                'lower': lower,
                'upper': upper
                })
            self.save_counterexample(i, counterexample)
            if self._verbosity > 0:
                print(f'image:{i}, class:{actual_label}, predcited:{predicted_label}, epsilon:{epsilon}')
        
        # generate dataframe and optionally save results to csv
        df = pd.DataFrame(data, columns=('image', 'class', 'predicted', 'epsilon', 'lower', 'upper'))
        self._results = set_df_dtypes(df, results_dtypes)
        if epsilons_outpath:
            create_output_path(epsilons_outpath)
            self._results.to_csv(epsilons_outpath)
        if counterexamples_outpath:
            create_output_path(counterexamples_outpath)
            with open(counterexamples_outpath, 'wb') as f:
                pickle.dump(self.counterexamples, f)
        return self
    
    def load_results(self, epsilons_path:str='', counterexamples_path:str='') -> ContextualRobustness:
        '''Load previously saved results

        Args:
            epsilons_path (str, optional): Path to results csv file. Defaults to ''.
            counterexamples_path (str, optional): Path to counterexamples pickle. Defaults to ''.

        Returns:
            ContextualRobustness: the object (self)
        '''
        if epsilons_path:
            self._results = set_df_dtypes(pd.read_csv(epsilons_path, index_col=0), results_dtypes)

        if counterexamples_path:
            with open(counterexamples_path, 'rb') as f:
                self._counterexamples = pickle.load(f)
        return self

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
        model_path='',
        model_name='',
        model_args=dict(),
        transform_fn=lambda x: x,
        transform_args=dict(),
        transform_name='',
        X=np.array([]),
        Y=np.array([]),
        sample_indexes=[],
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval'],
        marabou_options=dict(verbosity=defaults['marabou_verbosity']),
        verbosity=defaults['verbosity']
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
        ext = get_file_extension(model_path)
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
        ext = get_file_extension(self._model_path)
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
                print(f'evaluating image:{index}@epsilon:{guess}, label:{actual_label}, pred:{pred}')
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
            vals, stats = network.solve(options=Marabou.createOptions(**self._marabou_options), verbose=(self._verbosity > 3))
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

# ======================================================================
# ContextualRobustnessReporting
# ======================================================================
class ContextualRobustnessReporting:
    '''Class containing ContextualRobustness reporting functions'''
    @staticmethod
    def generate_epsilons_plot(
            cr: ContextualRobustness,
            outfile:str='epsilons.png',
            xlabel:str='',
            ylabel:str='epsilon',
            axis_fontsize:int=24,
            fontfamily:str='serif',
            fontweight:str='ultralight',
            usetex:bool=True
            ):
        '''Plots epsilons by class for a model/transform and saves as png

        Args:
            cr (ContextualRobustness): ContextualRobustness object
            outfile (str, optional): Output file path. Defaults to 'epsilons.png'.
            xlabel (str, optional): x axis label. Defaults to ''.
            ylabel (str, optional): y axis label. Defaults to 'epsilon'.
            axis_fontsize (int, optional): Fontsize for axis text. Defaults to 24.
            fontfamily (str, optional): Fontfamily for text. Defaults to 'serif'.
            fontweight (str, optional): Fontfamily for text. Defaults to 'ultralight'.
            usetex (bool, optional): Use latex for text. Defaults to True.
        '''
        plt.rc('text', usetex=usetex)
        plt.rc('font', family=fontfamily, weight=fontweight)
        ax = cr.results.boxplot(column=['epsilon'], by='class', return_type=None)
        fig = ax.get_figure()
        fig.suptitle('')
        plt.title('')
        plt.xlabel(xlabel, fontsize=axis_fontsize, weight=fontweight)
        plt.ylabel(ylabel, fontsize=axis_fontsize, weight=fontweight)
        ax.tick_params(axis='both', labelsize=axis_fontsize)

        create_output_path(outfile)
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f'saved epsilons plot to {outfile}')
    
    @staticmethod
    def generate_counterexamples_plot(
            cr: ContextualRobustness,
            outfile:str='./counterexamples.png',
            nrows:int=2,
            ncols:str='auto',
            figsize:tuple=(10, 10)
            ):
        '''Plots counterexamples for a model/transform and saves as png

        Args:
            cr (ContextualRobustness): ContextualRobustness object
            outfile (str, optional): Output file path. Defaults to './counterexamples.png'.
            nrows (int, optional): Number of rows. Defaults to 2.
            ncols (str, optional): Number of columns. Defaults to 'auto'.
            figsize (tuple, optional): Size of figure (w, h). Defaults to (10, 10).
        '''
        fig = plt.figure(figsize=figsize)
        ncols = len(cr.classes) if ncols == 'auto' else ncols
        gridImage = ImageGrid(
            fig,
            111,                        # similar to subplot(111)
            nrows_ncols=(nrows, ncols), # creates 2 x nClasses grid
            axes_pad=0.1,               # pad between axes in inch.
            share_all=True              # share x & y axes among subplots
            )
        gridImage[0].get_yaxis().set_ticks([])
        gridImage[0].get_xaxis().set_ticks([])
        X, _ = cr.dataset
        for c in cr.classes:
            # Placeholder (black square) used for classes where no results are present, 
            # or where no counterexample was found.
            x_orig, x_cex = np.zeros(cr.image_shape), np.zeros(cr.image_shape)
            sorted_df = cr.get_results(class_index=c, sort_by=['epsilon'])
            if sorted_df.shape[0] > 1:    
                # The 'test-based technique will always have a counterexample, however 
                # the formal technique may not. Find first sample with a counterexample 
                # nearest to the mean, and show placeholders for classes where no sample 
                # with a counterexample was found.
                mean_epsilon = np.mean(sorted_df.epsilon)
                upper_df = sorted_df[sorted_df.epsilon >= mean_epsilon]
                x_orig = X[upper_df['image'].iloc[0]]
                for idx in upper_df['image']:
                    if cr.get_counterexample(idx) is not None:
                        x_orig = X[idx]
                        x_cex = cr.get_counterexample(idx)
                        break
            gridImage[c].imshow(x_orig)
            gridImage[c + ncols].imshow(x_cex)

        plt.axis('off')
        create_output_path(outfile)
        fig.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f'saved counterexamples plot to {outfile}')
    
    @staticmethod
    def generate_class_accuracy_plot(
            cr:ContextualRobustness,
            outfile:str='./class-accuracy.png',
            axis_fontsize:int=12,
            legend_fontsize:int=14,
            fontfamily:str='serif',
            fontweight:str='ultralight',
            legend_loc:str='best',
            usetex:bool=True
            ):
        '''Plots accuracy of each class at various epsilons for a model/transform and saves as png

        Args:
            cr (ContextualRobustness): ContextualRobustness object
            outfile (str, optional): Output file path. Defaults to './counterexamples.png'.
            axis_fontsize (int, optional): Fontsize for axis text. Defaults to 12.
            legend_fontsize (int, optional): Fontsize for axis text. Defaults to 14.
            fontfamily (str, optional): Fontfamily for text. Defaults to 'serif'.
            fontweight (str, optional): Fontweight for text. Defaults to 'ultralight'.
            legend_loc (str, optional): Location of legend. Defaults to 'best'.
            usetex (bool, optional): Use latex for text. Defaults to True.
        '''
        plt.rc('text', usetex=usetex)
        plt.rc('font', family=fontfamily, weight=fontweight)
        # for each class, plot accuracy at different values of epsilon
        for c in cr.classes:
            class_total = cr.get_num_samples(class_index=c)
            # if no samples for the class
            if class_total == 0:
                continue
            class_accuracy = cr.get_accuracy(class_index=c)
            accuracy_report = [{
                'epsilon': 0,
                'accuracy': class_accuracy,
                'model': cr.model_name,
                'class': c
                }]
            sorted_class_df = cr.get_results(class_index=c, sort_by=['epsilon'])
            xSpace = np.linspace(1, 1000, 1000) / 1000
            for x in xSpace:
                sorted_class_df = sorted_class_df[sorted_class_df['epsilon'] >= x]
                correct = sorted_class_df.shape[0]
                if (correct>0):
                    accuracy_report.append({
                        'epsilon': x,
                        'accuracy': correct/class_total,
                        'model': cr.model_name,
                        'class' : c
                        })

            report_df = pd.DataFrame(accuracy_report, columns=('epsilon', 'accuracy'))
            plt.plot(report_df['epsilon'], report_df['accuracy'], label=f'class {c}')
        plt.grid()
        plt.legend(fontsize=legend_fontsize, loc=legend_loc)
        plt.tick_params(axis='both', labelsize=axis_fontsize)
        plt.xlabel('Epsilon', fontsize=axis_fontsize)
        plt.ylabel('Accuracy', fontsize=axis_fontsize)

        create_output_path(outfile)
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f'saved class accuracy report plot to {outfile}')
    
    @staticmethod
    def generate_accuracy_report_plot(
            cr_objects:typing.Sequence[ContextualRobustness],
            outfile:str='./accuracy-report.png',
            linestyles:tuple=(),
            axis_fontsize:int=12,
            legend_fontsize:int=14,
            fontfamily:str='serif',
            fontweight:str='ultralight',
            legend_loc:str='best',
            usetex:bool=True
            ):
        '''Plots epsilon/accuracy for a given transform on multiple models and saves as png

        Args:
            cr_objects (typing.Sequence[ContextualRobustness]): list of ContextualRobustness objects
            outfile (str, optional): Output file path. Defaults to './accuracy-report.png'.
            linestyles (tuple, optional): [description]. Defaults to ().
            axis_fontsize (int, optional): Fontsize for axis text. Defaults to 12.
            legend_fontsize (int, optional): Fontsize for axis text. Defaults to 14.
            fontfamily (str, optional): Fontfamily for text. Defaults to 'serif'.
            fontweight (str, optional): Fontweight for text. Defaults to 'ultralight'.
            legend_loc (str, optional): Location of legend. Defaults to 'best'.
            usetex (bool, optional): Use latex for text. Defaults to True.
        '''
        plt.rc('text', usetex=usetex)
        plt.rc('font', family=fontfamily, weight=fontweight)
        for i,cr in enumerate(cr_objects):
            model_name = cr.model_name if len(cr.model_name) > 0 else f'model{i}'
            total = cr.num_samples
            accuracy = cr.accuracy
            accuracy_data = [{
                    'epsilon': 0,
                    'accuracy': accuracy,
                    'model': cr.model_name
                    }]
            sorted_epsilons = cr.get_results(sort_by=['epsilon'])
            x_space = np.linspace(1, 1000, 1000) / 1000
            for x in x_space:
                sorted_epsilons = sorted_epsilons[sorted_epsilons['epsilon'] >= x]
                correct = sorted_epsilons.shape[0]
                if correct > 0:
                    accuracy_data.append(dict(epsilon=x, accuracy=correct/total, model=model_name))
            accuracy_df = pd.DataFrame(accuracy_data, columns=('epsilon', 'accuracy'))
            ls = linestyles[i] if len(linestyles) > i else '-'
            plt.plot(
                accuracy_df['epsilon'],
                accuracy_df['accuracy'],
                label=model_name,
                linestyle=ls
                )
        plt.grid()
        plt.legend(fontsize=legend_fontsize, loc=legend_loc)
        plt.tick_params(axis='both', labelsize=axis_fontsize)
        plt.xlabel('Epsilon', fontsize=axis_fontsize)
        plt.ylabel('Accuracy', fontsize=axis_fontsize)
        create_output_path(outfile)
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f'saved accuracy report plot to {outfile}')
