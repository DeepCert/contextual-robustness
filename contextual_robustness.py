import sys, enum, copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import create_output_path, softargmax
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
    eps_interval=0.002
    )

# ======================================================================
# BaseContextualRobustness
# ======================================================================
class BaseContextualRobustness(metaclass=ABCMeta):
    '''
    Base-class for ContextualRobustness subclasses; Implements common functionality, 
    properties, and defines abstract methods.

    Properties:
    '''
    def __init__(
        self,
        model='',
        model_name='',
        transform_fn=lambda x, epsilon: x,
        transform_args=dict(),
        transform_name='',
        X=np.array([]),
        Y=np.array([]),
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval']
        ):
        assert len(model) > 0, 'model is required'
        assert X.shape[0] == Y.shape[0], 'X and Y must have equal number of items'
        assert callable(transform_fn), 'transform_fn must be callable (e.g. a function)'
        
        self._model = self._load_model(model)
        self._model_name = model_name
        self._transform_fn = transform_fn
        self._transform_args = transform_args
        self._transform_name = transform_name
        self._X, self._Y = X, Y
        self._eps_lower = eps_lower
        self._eps_upper = eps_upper
        self._eps_interval = eps_interval

        # find indexes of correctly predicted samples
        self._correct_sample_indexes = self._find_correct_sample_indexes(X, Y)
        print(f'filtered {len(X) - len(self._correct_sample_indexes)} incorrectly predicted samples')
        
        # compute model accuracy
        self._accuracy = len(self._correct_sample_indexes) / len(X)
        print(f'model accuracy is {self.accuracy}')

    @property
    @abstractmethod
    def technique(self):
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
    
    def get_num_samples(self, class_index=None):
        '''
        returns number of samples in dataset (optionally for a single class).
        
        Parameters:
            class_index (integer) - when specified, returns number of samples for the specified class
        
        Returns:
            integer
        '''
        if class_index is not None:
            return len([y for y in self.dataset[1] if np.argmax(y) == class_index])
        return self.dataset[0].shape[0]
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
            sample_indexes = [i for i in range(self._Y.shape[0]) if np.argmax(self._Y[i]) == class_index]
            correct_sample_indexes = [i for i in self._correct_sample_indexes if np.argmax(self._Y[i]) == class_index]
            return len(correct_sample_indexes) / len(sample_indexes)
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
    def _load_model(self, model):
        '''
        Called by constructor to load a model (or MarabouNetwork). Abstract method which must 
        be implemented by subclasses.

        Arguments:
            model (any) - the model
        '''
        return model

    @abstractmethod
    def _find_epsilon(self, x, y, index=None):
        '''
        Finds epsilon for a given image. Abstract method which must be implemented by subclasses.

        Arguments:
            x     (np.array) - (*required) the image
            y     (np.array) - (*required) label for image (x)
            index (integer)  - index of image (x)
        '''
        pass

    @abstractmethod
    def _find_correct_sample_indexes(self, X, Y):
        '''
        Finds the indexes of all correctly predicted samples in X. Abstract method which must be 
        implemented by subclasses.

        Arguments:
            X     (np.array) - (*required) images
            Y     (np.array) - (*required) labels for images (X)
        '''
        pass
    
    @abstractmethod
    def transform_image(self, x, epsilon):
        '''
        Transforms images. Abstract method which must be 
        implemented by subclasses.

        Arguments:
            X     (np.array) - (*required) images
            Y     (np.array) - (*required) labels for images (X)
        '''
        pass

    def analyze(self, outfile='./epsilons.csv', save_csv=True, verbose=0):
        '''
        tests all correctly predicted samples, and optionally stores the results in a csv
        
        Parameters:
            outfile (string)  - output csv file path
            save_csv (bool)   - enables/disables writing to csv (default=True)
            verbose (integer) - increases verbosity of console output (0 or 1)
        
        Returns:
            ContextualRobustness object
        '''
        print(f'analyzing {self.transform_name} on {len(self._correct_sample_indexes)} samples. this may take some time...')
        data = []
        for i in self._correct_sample_indexes:
            x, y = self._X[i], self._Y[i]
            actual_label = np.argmax(y)
            lower, upper, epsilon, predicted_label = self._find_epsilon(x, y, index=i)
            data.append({
                'image': i,
                'class': actual_label,
                'predicted': predicted_label,
                'epsilon': epsilon,
                'lower': lower,
                'upper': upper
                })
            if verbose:
                print(f'image:{i}, class:{actual_label}, predcited:{predicted_label}, epsilon:{epsilon}')
        
        # generate dataframe and optionally save results to csv
        self._results = pd.DataFrame(data, columns=('image','class','predicted','epsilon', 'lower', 'upper'))
        if save_csv:
            create_output_path(outfile)
            self._results.to_csv(outfile)
        return self
    
    def load_results(self, csv_path):
        '''
        loads saved results from csv file
        
        Parameters:
            csv_path (string) - (*required) path to the csv containing results
        
        Returns:
            ContextualRobustness object
        '''
        self._results = pd.read_csv(csv_path)
        return self

# ======================================================================
# ContextualRobustnessTest
# ======================================================================
class ContextualRobustnessTest(BaseContextualRobustness):
    '''
    ContextualRobustness class for test-based technique
    '''
    def __init__(
        self,
        model = '',
        model_name='',
        X=np.array([]),
        Y=np.array([]),
        transform_fn=lambda x, epsilon: x,
        transform_args=dict(),
        transform_name='',
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval']
        ):
        assert bool(model), 'model is required'
        assert X.shape[0] == Y.shape[0], 'X and Y must have equal number of items'
        assert callable(transform_fn), 'transform_fn must be callable (e.g. a function)'

        super().__init__(
            model=model,
            model_name=model_name,
            X=X,
            Y=Y,
            transform_fn=transform_fn,
            transform_args=transform_args,
            transform_name=transform_name,
            eps_lower=eps_lower,
            eps_upper=eps_upper
            )
    
    @property
    def technique(self):
        return Techniques.TEST

    def _load_model(self, model):
        '''
        loads a tensorflow model

        Parameters:
            model (string) - (*required) path to tensorflow model
        '''
        return tf.keras.models.load_model(model)

    def _find_correct_sample_indexes(self, X, Y):
        '''
        returns list of indexes of correctly predicted samples
        
        Parameters:
            X (np.array) - (*required) input images
            Y (np.array) - (*required) labels for X
        
        Returns:
            list
        '''
        Y_p = self._model.predict(X)
        return [i for i in range(len(X)) if np.argmax(Y_p[i]) == np.argmax(Y[i])]
    
    def _find_epsilon(self, x, y, index=None):
        '''
        finds the epsilon value of the transform_fn applied to x
        
        Parameters:
            x     (np.array) - (*required) input image
            y     (np.array) - (*required) label for x
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
        while ((upper - lower) > interval):
            guess = lower + (upper - lower) / 2.0
            x_trans = self.transform_image(x, guess)
            pred = np.argmax(self._model.predict(x_trans.reshape((1,) + x_trans.shape)))
            if pred == actual_label:
                # correct prediction
                lower = guess
            else:
                # incorrect prediction
                upper = guess
                predicted_label = pred
                epsilon = guess
        return lower, upper, epsilon, predicted_label
    
    def transform_image(self, x, epsilon):
        '''
        applies transform_fn to x with value of epsilon
        
        Parameters:
            x       (np.array) - (*required) input image
            epsilon (float)    - (*required) amount of transform to apply to x
        
        Returns:
            np.array - transformed image
        '''
        return self._transform_fn(x, epsilon=epsilon, **self._transform_args)

# ======================================================================
# ContextualRobustnessFormal
# ======================================================================
class ContextualRobustnessFormal(metaclass=BaseContextualRobustness):
    '''
    ContextualRobustness class for formal verification technique
    '''
    def __init__(
        self,
        model=None,
        model_name='',
        transform_fn=lambda x, epsilon, output_index: x,
        transform_args=dict(),
        transform_name='',
        X=np.array([]),
        Y=np.array([]),
        eps_lower=defaults['eps_lower'],
        eps_upper=defaults['eps_upper'],
        eps_interval=defaults['eps_interval']
        ):
        super().__init__(
            model=model,
            model_name=model_name,
            X=X,
            Y=Y,
            transform_fn=transform_fn,
            transform_args=transform_args,
            transform_name=transform_name,
            eps_lower=eps_lower,
            eps_upper=eps_upper
            )
        self._counterexamples = {}
    
    @property
    def technique(self):
        return Techniques.FORMAL

    def _load_model(self, model):
        '''
        Loads a MarabouNetwork object. Saves a copy of the object.

        Parameters:
            model (MarabouNetwork) - the marabou network.
        
        Returns:
            model (MarabouNetwork)
        '''
        return copy.deepcopy(model)
    
    def _copy_model(self):
        '''
        Makes a copy of the original saved model.

        Returns:
            model (MarbouNetwork)
        '''
        return copy.deepcopy(self._model)
    
    def _find_correct_sample_indexes(self, X, Y):
        '''
        Returns list of indexes of correctly predicted samples
        
        Parameters:
            X (np.array) - (*required) input images
            Y (np.array) - (*required) labels for X
        
        Returns:
            list
        '''
        Y_p = np.array([softargmax(self._model.evaluate(x)) for x in X])
        return [i for i in range(len(X)) if np.argmax(Y_p[i]) == np.argmax(Y[i])]
    
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
        while ((upper - lower) > interval):
            guess = lower + (upper - lower) / 2.0
            verified = True
            for output in range(len(self.classes)):
                if actual_label == output:
                    continue
                network = self.transform_image(x, guess, output)
                result, code = network.solve(options=Marabou.createOptions(solveWithMILP=True))
                if code == 'SAT':
                    self._counterexamples.append({
                        'x_index': index,
                        'predicted_label':output,
                        'epsilon': guess,
                        'result':result,
                        'query': network.getMarabouQuery()
                        })
                    pred = output
                    verified = False
                    break
                elif code == 'UNSAT':
                    continue
                else:
                    # handle error
                    verified = False
                    print(code)
                    assert(False)
            if verified:
                # correct prediction
                lower = guess
            else:
                # incorrect prediction
                upper = guess
                predicted_label = pred
                epsilon = guess
        return lower, upper, epsilon, predicted_label
    
    def transform_image(self, x, epsilon, output_index):
        '''
        applies transform_fn to the network to encode the image transform for an, 
        image, epsilon, and output_index as a marabou query.
        
        Parameters:
            x            (np.array) - (*required) input image
            epsilon      (float)    - (*required) amount of transform to apply to x
            output_index (integer)  - (*required) amount of transform to apply to x
        
        Returns:
            (MarabouNetwork) - the network with image encoded as a Marabou input query
        '''
        # create a copy of the original model
        network = self._copy_model()
        # encode the transform as a marabou input query using the transform_fn
        network = self._transform_fn(network, x, epsilon, output_index, **self._transform_args)
        return network

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
            upper = 100
            sorted_df = cr.get_results(class_index=c, sort_by=['epsilon'])
            mean_epsilon = np.mean(sorted_df.epsilon)
            upper_df = sorted_df[sorted_df.epsilon >= mean_epsilon]
            idx = upper_df['image'].iloc[0]
            # upper = upper_df['upper'].iloc[0]
            epsilon = upper_df['epsilon'].iloc[0]
            gridImage[c].imshow(X[idx])
            # get the transformed image to display
            if cr.technique == Techniques.FORMAL:
                # TODO: get transformed image from formal verification counterexamples
                transformed_image = None
            else:
                transformed_image = cr.transform_image(X[idx], epsilon=epsilon)
            gridImage[c + ncols].imshow(transformed_image)
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
            class_accuracy = cr.get_accuracy(class_index=c)
            accuracy_report = [{
                'epsilon': 0,
                'accuracy': class_accuracy,
                'model': cr.model_name,
                'class': c
                }]
            class_total = cr.get_num_samples(class_index=c)
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
