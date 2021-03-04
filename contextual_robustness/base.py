import enum, pickle, typing
import numpy as np
import pandas as pd
import tensorflow as tf
from contextual_robustness.utils import set_df_dtypes, create_output_path
from abc import ABCMeta, abstractmethod

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
