import os
import tensorflow as tf
from contextual_robustness import ContextualRobustnessTest, ContextualRobustnessReporting
from transforms import test_transforms as transforms
from load_data import load_cifar

# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(outdir, image_formats):
    # load dataset
    _, _, X_test, Y_test, _ = load_cifar()

    # define models
    models = ('4a', '4b', '5a', '5b', '6a', '6b')

    # generate plots for each model/transform combo
    cr_objects = []
    for transform in transforms.keys():
        transform_name = transform.capitalize()
        for m in models:
            model_name = f'Model {m}'
            print(f'{("-"*80)}\nGenerating plots for {model_name} {transform_name}\n{("-"*80)}')
            # load model
            model = tf.keras.models.load_model(f'./models/cifar/model{m}.h5')
            # instantiate ContextualRobustness object
            cr = ContextualRobustnessTest(
                model=model,
                model_name=model_name,
                X=X_test,
                Y=Y_test,
                transform_fn=transforms[transform]['fn'],
                transform_args=transforms[transform]['args'],
                transform_name=transform_name
                )
            # load results from csv
            cr.load_results(
                csv_path=os.path.join('./results/cifar/test/data', f'model{m}-{transform}.csv')
                )
            cr_objects.append(cr)
            
            # generate plots for the model/transform
            for image_format in image_formats:
                out_dir = os.path.join(outdir, f'images/{image_format}')
                ContextualRobustnessReporting.generate_epsilons_plot(
                    cr,
                    outfile=os.path.join(out_dir, f'model{m}-{transform}_epsilons.{image_format}')
                    )
                ContextualRobustnessReporting.generate_counterexamples_plot(
                    cr,
                    outfile=os.path.join(out_dir, f'model{m}-{transform}_counterexamples.{image_format}')
                    )
                ContextualRobustnessReporting.generate_class_accuracy_plot(
                    cr,
                    outfile=os.path.join(out_dir, f'model{m}-{transform}_class-accuracy.{image_format}'),
                    legend_fontsize=16,
                    axis_fontsize=20,
                    legend_loc='lower left'
                    )

    # generate accuracy reports comparing all models on each transform
    linestyles = ('--','--', '-', '-', ':', ':')
    for transform in transforms.keys():
        transform_name = transform.capitalize()
        transform_cr_objects = [cr for cr in cr_objects if cr.transform_name.lower() == transform]
        print(f'{("-"*80)}\nGenerating {transform_name} accuracy report for models {", ".join(models)}\n{("-"*80)}')
        for image_format in image_formats:
            ContextualRobustnessReporting.generate_accuracy_report_plot(
                transform_cr_objects,
                outfile=os.path.join(outdir, f'{image_format}/cifar-{transform}_accuracy.{image_format}'),
                linestyles=linestyles,
                legend_fontsize=16,
                axis_fontsize=20
                )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir',
        default='./results/cifar/test/images',
        help='output directory')
    parser.add_argument('-f', '--formats',
        nargs='*',
        default=['png'],
        help='image format(s) (png and/or pdf)')
    args = parser.parse_args()
    
    main(args.outdir, args.formats)