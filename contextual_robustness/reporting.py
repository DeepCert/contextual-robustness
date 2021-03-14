import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from contextual_robustness.base import ContextualRobustness, Techniques
from contextual_robustness.utils import _create_output_path, resize_image
from contextual_robustness.datasets import _load_placeholder_images

PLACEHOLDERS = _load_placeholder_images()
NO_CEX_IMG = PLACEHOLDERS.get('no_cex')
NO_IMAGE_IMG = PLACEHOLDERS.get('no_image')

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

        _create_output_path(outfile)
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f'saved epsilons plot to {outfile}')

    @staticmethod
    def generate_counterexamples_plot(
            cr: ContextualRobustness,
            outfile:str='./counterexamples.png',
            dpi:int=144,
            show_labels:bool=True,
            label_fontsize:int=12,
            fontfamily:str='serif',
            fontweight:str='ultralight',
            usetex:bool=True
            ):
        '''Plots counterexamples for a model/transform and saves as png

        Args:
            cr (ContextualRobustness): ContextualRobustness object
            outfile (str, optional): Output file path. Defaults to './counterexamples.png'.
            dpi (int, optional): Dots per inch. Defaults to 144.
            show_labels (bool, optional): Show the class labels? Defaults to True.
            label_fontsize (int, optional): Fontsize of class labels. Defaults to 12.
            fontfamily (str, optional): Fontfamily for text. Defaults to 'serif'.
            fontweight (str, optional): Fontfamily for text. Defaults to 'ultralight'.
            usetex (bool, optional): Use latex for text. Defaults to True.
        '''
        # configure font
        plt.rc('text', usetex=usetex)
        plt.rc('font', family=fontfamily, weight=fontweight)

        nrows, ncols = 2, len(cr.classes)
        images = [None] * (nrows * ncols)
        X, _ = cr.dataset

        for c in cr.classes:
            # no_image placeholder for classes where images/results are not present (test or formal).
            # no_cex placeholder for classes where no counterexample found (formal only).
            x_orig, x_cex = NO_IMAGE_IMG, NO_CEX_IMG
            sorted_df = cr.get_results(class_index=c, sort_by=['epsilon'])
            if sorted_df.shape[0] > 1:

                mean_epsilon = np.mean(sorted_df.epsilon)
                upper_df = sorted_df[sorted_df.epsilon >= mean_epsilon]
                row = upper_df.iloc[0]
                x_idx = row['image'].astype(int)
                epsilon = row['epsilon']
                x_orig = X[x_idx]
                if cr.technique == Techniques.TEST:
                    # for test technique, generate counterexample if not saved
                    if cr.get_counterexample(x_idx) is not None:
                        x_cex = cr.get_counterexample(x_idx)
                    else:
                        x_cex = cr.transform_image(x_orig, epsilon)
                elif cr.technique == Techniques.FORMAL:
                    # for formal technique, counterexample must be read from results
                    for idx in upper_df['image'].astype(int):
                        if cr.get_counterexample(idx) is not None:
                            x_orig = X[idx]
                            x_cex = cr.get_counterexample(idx)
                            break

            images[c] = dict(title=f'class {c}', image=x_orig)
            images[c + ncols] = dict(title=f'', image=x_cex)

        # create the figure & add images
        figure, ax = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(ncols, nrows),
            gridspec_kw=dict(wspace=0.06, hspace=0.03)
            )
        for i,image in enumerate(images):
            ax.ravel()[i].imshow(image['image'])
            # only show image title (class #) on top row
            if show_labels and i < ncols:
                ax.ravel()[i].set_title(image['title'], fontsize=label_fontsize)
            # configure axes & hide ticks
            plt.setp(ax.ravel()[i].spines.values(), linewidth=0.5)
            ax.ravel()[i].get_xaxis().set_ticks([])
            ax.ravel()[i].get_yaxis().set_ticks([])
        # setup output path and save figure to outfile
        _create_output_path(outfile)
        figure.savefig(outfile, bbox_inches='tight', dpi=dpi)
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

        _create_output_path(outfile)
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
        _create_output_path(outfile)
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f'saved accuracy report plot to {outfile}')
