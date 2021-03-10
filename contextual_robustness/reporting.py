import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from contextual_robustness.base import ContextualRobustness
from contextual_robustness.utils import create_output_path, resize_image
from contextual_robustness.datasets import load_placeholder_images

PLACEHOLDERS = load_placeholder_images()
NO_CEX_IMG = PLACEHOLDERS.get('no_cex')
NO_IMAGE_IMG = PLACEHOLDERS.get('no_image')

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
            # no_image placeholder used for classes where no images/results are present.
            # no_cex placeholder used for classes where image was present, but no counterexample found.
            x_orig, x_cex = resize_image(NO_IMAGE_IMG, cr.image_size), resize_image(NO_CEX_IMG, cr.image_size)
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
