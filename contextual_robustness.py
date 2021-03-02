import sys, enum, copy, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import set_df_dtypes, create_output_path, softargmax, get_file_extension
from abc import ABCMeta, abstractmethod

# TODO: remove sys.path.append after maraboupy pip package is available.
sys.path.append('../Marabou/')
from maraboupy import Marabou

class Techniques(enum.Enum):
    TEST='test'
    FORMAL='formal'

defaults = dict(
    eps_lower=0.0,
    eps_upper=1.0,
    eps_interval=0.002,
    verbosity=0,
    marabou_verbosity=0
    )

results_dtypes = {
    'image': np.int64,
    'class': np.int64,
    'predicted': np.int64,
    'epsilon': np.float64,
    'upper': np.float64,
    'lower': np.float64
    }

# ======================================================================
# BaseContextualRobustness
# ======================================================================
class BaseContextualRobustness(metaclass=ABCMeta):
    '''
    Base-class for ContextualRobustness subclasses; Implements common functionality, 
    properties, and defines abstract methods.
    '''
    def __init__(
        self,
        model_path='',
        model_name='',
        transform_fn=lambda x, epsilon: x,
        transform_args=dict(),
        transform_name='',
        X=np.array([]),
        Y=np.array([]),
        sample_indexes=[],
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval'],
        verbosity=defaults['verbosity'],
        ):
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
        # self._accuracy = len(self._correct_sample_indexes) / len(X)
        self._accuracy = len(self._correct_sample_indexes) / len(self._sample_indexes)
        print(f'accuracy is {self.accuracy}')
        # examples of images @ epsilon where network's prediction changed
        self._counterexamples = dict()

    @property
    @abstractmethod
    def technique(self):
        '''
        Returns a property from Techniques Enum. Abstract method implemented by subclasses.
        '''
        return None

    @property
    def model_name(self):
        return self._model_name
    
    @property
    def transform_name(self):
        return self._transform_name
    
    @property
    def classes(self):
        ''' returns list of classes '''
        return sorted(np.unique([np.argmax(self._Y[i]) for i in range(self._Y.shape[0])]))
    
    @property
    def dataset(self):
        ''' returns tuple containing the dataset (X, Y) '''
        return self._X, self._Y
    
    @property
    def image_shape(self):
        ''' returns shape of images '''
        return self.dataset[0].shape[1:]
    
    @property
    def n_pixels(self):
        ''' returns shape of images '''
        prod = 1
        for dim in self.image_shape:
            prod *= dim
        return prod

    @property
    def counterexamples(self):
        return self._counterexamples

    def get_counterexample(self, x_index):
        return self.counterexamples.get(f'image{x_index}')
    
    def save_counterexample(self, x_index, counterexample):
        self._counterexamples[f'image{x_index}'] = counterexample
    
    def get_num_samples(self, class_index=None):
        '''
        returns number of samples in dataset (optionally for a single class).
        
        Parameters:
            class_index (integer) - when specified, returns number of samples for the specified class
        
        Returns:
            integer
        '''
        if class_index is not None:
            return len([si for si in self._sample_indexes if np.argmax(self.dataset[1][si]) == class_index])
        return len(self._sample_indexes)
    num_samples = property(get_num_samples)

    def get_accuracy(self, class_index=None):
        '''
        returns accuracy of model (optionally for a single class)
        
        Parameters:
            class_index (integer) - when specified, returns accuracy for a single class.
        
        Returns:
            float
        '''
        if class_index is not None:
            sample_indexes = [si for si in self._sample_indexes if np.argmax(self._Y[si]) == class_index]
            correct_sample_indexes = [i for i in self._correct_sample_indexes if np.argmax(self._Y[i]) == class_index]
            return len(correct_sample_indexes) / len(sample_indexes) if len(sample_indexes) > 0 else 0
        return self._accuracy
    accuracy = property(get_accuracy)

    def get_results(self, class_index=None, sort_by=[]):
        '''
        returns a dataframe containing the analysis results (optionally for a single class and/or sorted)
        
        Parameters:
            class_index (integer) - when specified, returns results for a single class.
            sort_by     (list)    - when specified, sorts results by the specified field(s)
        
        Returns:
            pd.DataFrame
        '''
        results = self._results
        if class_index is not None:
            results = results[results['class'] == class_index]
        if len(sort_by) > 0:
            results = results.sort_values(by=sort_by)
        return results
    results = property(get_results)

    @abstractmethod
    def _find_epsilon(self, x, y, index=None):
        '''
        Finds epsilon for a given image; Abstract method which must be implemented by subclasses.

        Arguments:
            x     (np.array) - (*required) the image
            y     (np.array) - (*required) label for image (x)
            index (integer)  - index of image (x)
        
        Returns:
            (tuple) - lower, upper, epsilon, predicted_label, counterexample

            lower           : last lower epsilon from binary search
            upper           : last upper epsilon from binary search
            epsilon         : last upper epsilon from binary search
            predicted_label : upper epsilon from binary search
            counterexample  : image at epsilon where prediction changed
        '''
        pass

    @abstractmethod
    def _find_correct_sample_indexes(self, X, Y):
        '''
        Returns list of indexes of correctly predicted samples; Abstract method implemented by subclasses
        '''
        pass
    
    @abstractmethod
    def _load_model(self, model_path):
        '''
        Loads a model; Abstract method implemented by subclasses
        '''
        pass
    
    def analyze(self, epsilons_outpath='./epsilons.csv', counterexamples_outpath='./counterexamples.p'):
        '''
        Tests all correctly predicted samples, and optionally stores the results in a csv
        
        Parameters:
            epsilons_outpath        (string)  - epsilons output csv file path
            counterexamples_outpath (string)  - counterexamples pickle path
        
        Returns:
            ContextualRobustness object
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
    
    def load_results(self, epsilons_path='', counterexamples_path=''):
        '''
        Loads saved results from csv file
        
        Parameters:
            epsilons_path        (string) - path to the csv containing epsilons
            counterexamples_path (string) - path to the pickle containing counterexamples

        Returns:
            ContextualRobustness object
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
class ContextualRobustnessTest(BaseContextualRobustness):
    '''
    ContextualRobustness class for test-based technique

    Parameters:
        model_path     (path)     - (*required) path to saved model
        model_name     (string)   - name of model
        X              (np.array) - (*required) images
        Y              (np.array) - (*required) labels for images
        sample_indexes ([int])    - indexes of samples to test
        transform_fn   (function) - (*required) transform function
        transform_args (dict)     - extra args for transform function
        transform_name (string)   - name of transform
        eps_lower      (float)    - lower bound for epsilon
        eps_upper      (float)    - upper bound for epsilon
        eps_interval   (float)    - interval between epsilons
        verbosity       (int)     - amount of logging (0-4)
    '''
    def __init__(
        self,
        model_path = '',
        model_name='',
        X=np.array([]),
        Y=np.array([]),
        sample_indexes=[],
        transform_fn=lambda x, epsilon: x,
        transform_args=dict(),
        transform_name='',
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval'],
        verbosity=defaults['verbosity']
        ):
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
    def technique(self):
        return Techniques.TEST
    
    def _load_model(self, model_path):
        '''
        loads a tensorflow model
        
        Parameters:
            model_path (string) - path to tensorflow model
        
        Returns:
            tensorflow model
        '''
        return tf.keras.models.load_model(model_path)
    
    def _find_correct_sample_indexes(self, X, Y):
        '''
        returns list of indexes of correctly predicted samples
        
        Parameters:
            X (np.array) - (*required) input images
            Y (np.array) - (*required) labels for X
        
        Returns:
            list
        '''
        Y_p = self._model.predict(np.array([X[si] for si in self._sample_indexes]))
        return [si for i,si in enumerate(self._sample_indexes) if np.argmax(Y_p[i]) == np.argmax(Y[si])]
    
    def _find_epsilon(self, x, y, index=None):
        '''
        finds the epsilon value of the transform_fn applied to x
        
        Parameters:
            x     (np.array) - (*required) input image
            y     (np.array) - (*required) label for x
            index (integer)  - index of image (just for logging)
        
        Returns:
            tuple (lower, upper, epsilon, predicted_label)
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
    
    def transform_image(self, x, epsilon):
        return self._transform_fn(x, epsilon=epsilon, **self._transform_args)

# ======================================================================
# ContextualRobustnessFormal
# ======================================================================
class ContextualRobustnessFormal(BaseContextualRobustness):
    '''
    ContextualRobustness class for formal verification technique

    Parameters:
        model_path      (string)   - (*required) path to the model loaded as MarabouNetwork
        model_name      (string)   - name of model
        model_args      (dict)     - args passed to Marabou.read_* when loading network (https://neuralnetworkverification.github.io/Marabou/API/0_Marabou.html)
        X               (np.array) - (*required) images
        Y               (np.array) - (*required) labels for images
        sample_indexes  ([int])    - list of specific sample indexes to test
        transform_fn    (function) - (*required) transform encoding function
        transform_args  (dict)     - extra args for transform function
        transform_name  (dict)     - name of the transform
        eps_lower       (float)    - lower bound for epsilon
        eps_upper       (float)    - upper bound for epsilon
        eps_interval    (float)    - interval between epsilons
        marabou_options (dict)     - options passed to Marabou's 'solve' function
        verbosity       (int)      - amount of logging (0-4)
    '''
    def __init__(
        self,
        model_path='',
        model_name='',
        model_args=dict(),
        transform_fn=lambda x, epsilon, output_index: x,
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
        ):
        self._model_args = model_args
        self._marabou_options = marabou_options
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
    def technique(self):
        return Techniques.FORMAL
    
    def _load_model(self, model_path):
        '''
        Loads model as a MarabouNetwork object

        Parameters:
            model_path (string) - model to load (NNet, Tensorflow (pb), HDF5, or ONNX)
        
        Returns:
            (MarabouNetwork) - the MarabouNetwork object
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
    
    def _find_correct_sample_indexes(self, X, Y):
        '''
        returns list of indexes of correctly predicted samples
        
        Parameters:
            X (np.array) - (*required) input images
            Y (np.array) - (*required) labels for X
        
        Returns:
            list
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
    
    def _find_epsilon(self, x, y, index=None):
        '''
        Finds the epsilon value of the transform_fn applied to x using formal verification,
        and saves counterexamples returned by Marabou.
        
        Parameters:
            x     (np.array) - (*required) input image
            y     (np.array) - (*required) label for image (x)
            index (integer)  - index of image (x)
        
        Returns:
            tuple (lower, upper, epsilon, predicted_label)
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
    
    def _find_counterexample(self, x, y, epsilon, x_index=None):
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
    '''
    Report generating functions for ContextualRobustness results objects.

    Functions:
        generate_epsilons_plot(cr, outfile)            - plots epsilons by class for a model/transform
        generate_counterexamples_plot(cr, outfile)     - plots counterexamples by class for a model/transform
        generate_class_accuracy_plot(cr, outfile)      - plots epsilon/accuracy for a single model/transform
        generate_accuracy_report_plot(crobjs, outfile) - plots epsilon/accuracy for multiple models/transforms
    '''
    @staticmethod
    def generate_epsilons_plot(
            cr,
            outfile='epsilons.png',
            xlabel='',
            ylabel='epsilon',
            axis_fontsize=24,
            fontfamily='serif',
            fontweight='ultralight',
            usetex=True
            ):
        '''
        plots epsilons by class for a model/transform and saves as png

        Parameters:
            cr            (ContextualRobustness*) - (*required) ContextualRobustness object
            outfile       (string)                - output file path
            xlabel        (string)                - x axis label
            ylabel        (string)                - y axis label
            axis_fontsize (integer)               - fontsize for axis text
            fontweight    (string)                - fontweight for plot text
            fontfamily    (string)                - fontfamily for plot text
            usetex        (bool)                  - use latex for text
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
            cr,
            outfile='./counterexamples.png',
            nrows=2,
            ncols='auto',
            figsize=(10, 10)
            ):
        '''
        plots counterexamples for a model/transform and saves as png

        Parameters:
            cr      (ContextualRobustness*) - (*required) ContextualRobustness object
            outfile (string)                - output file path
            nrows   (integer)               - number of rows
            ncols   (integer)               - number of columns (default='auto')
            figsize (tuple)                 - size of figure (w, h)
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
                for idx in upper_df['image']:
                    if cr.get_counterexample(idx) is not None:
                        x_orig = X[idx]
                        x_cex = cr.get_counterexample(idx)
                        print('XORIG:', x_orig)
                        print('XCEX:', x_cex)
                        gridImage[c + ncols].imshow(x_cex)
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
            cr,
            outfile='./class-accuracy.png',
            axis_fontsize=12,
            legend_fontsize=14,
            fontfamily='serif',
            fontweight='ultralight',
            legend_loc='best',
            usetex=True
            ):
        '''
        plots accuracy of each class at various epsilons for a model/transform and saves as png

        Parameters:
            cr              (ContextualRobustness*) - (*required) ContextualRobustness object
            outfile         (string)                - output file path
            axis_fontsize   (integer)               - fontsize for axis text
            legend_fontsize (integer)               - fontsize for legend text
            fontweight      (string)                - fontweight for plot text
            fontfamily      (string)                - fontfamily for plot text
            legend_loc      (string)                - location of legend (best, lower left, upper right, ...)
            usetex          (bool)                  - use latex for text
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
            cr_objects,
            outfile='./accuracy-report.png',
            linestyles=(),
            axis_fontsize=12,
            legend_fontsize=14,
            fontfamily='serif',
            fontweight='ultralight',
            legend_loc='best',
            usetex=True
            ):
        '''
        plots epsilon/accuracy for a given transform on multiple models and saves as png

        Parameters:
            cr_objects      ([ContextualRobustness*]) - (*required) list of ContextualRobustness objects
            outfile         (string)                  - output file path
            axis_fontsize   (integer)                 - fontsize for axis text
            legend_fontsize (integer)                 - fontsize for legend text
            fontweight      (string)                  - fontweight for plot text
            fontfamily      (string)                  - fontfamily for plot text
            linestyles      (tuple)                   - linestyle for each model (matplotlib syntax)
            legend_loc      (string)                  - location of legend (best, lower left, upper right, ...)
            usetex          (bool)                    - use latex for text
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
