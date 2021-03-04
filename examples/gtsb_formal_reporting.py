#!./venv/bin/python
import os
from contextual_robustness import ContextualRobustnessFormal, ContextualRobustnessReporting
from contextual_robustness.transforms import formal_transforms as transforms
from contextual_robustness.load_data import load_gtsrb
from contextual_robustness.utils import parse_indexes, remove_softmax_activation

# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(outdir, sample_indexes, image_formats):
    # load dataset
    _, _, X_test, Y_test, _ = load_gtsrb()

    # define transforms & models
    models = ('4a', '4b')

    # analyze each model on each transform
    for transform in transforms.keys():
        transform_name = transform.capitalize()
        for m in models:
            model_name = f'Model{m}'
            model_path = f'./models/gtsb/model{m}-verification'

            # create a copy of the h5 model without softmax activation
            remove_softmax_activation(
                f'./models/gtsb/model{m}.h5',
                save_path=model_path
                )

            print(f'{("-"*80)}\nGenerating plots for {model_name} {transform_name}\n{("-"*80)}')
            cr = ContextualRobustnessFormal(
                model_path=model_path,
                model_name=model_name,
                model_args=dict(modelType='savedModel_v2'),
                X=X_test,
                Y=Y_test,
                transform_fn=transforms[transform]['fn'],
                transform_args=transforms[transform]['args'],
                transform_name=transform_name,
                sample_indexes=sample_indexes
                )
            cr.load_results(
                epsilons_path=os.path.join('./results/gtsb/formal/data', f'model{m}-{transform}.csv'),
                counterexamples_path=os.path.join('./results/gtsb/formal/data', f'model{m}-{transform}-counterexamples.p')
                )
            
            # generate plots for the model/transform
            for image_format in image_formats:
                out_dir = os.path.join(outdir, f'{image_format}')
                ContextualRobustnessReporting.generate_epsilons_plot(
                    cr,
                    outfile=os.path.join(out_dir, f'model{m}-{transform}_epsilons.{image_format}')
                    )
                ContextualRobustnessReporting.generate_counterexamples_plot(
                    cr,
                    outfile=os.path.join(out_dir, f'model{m}-{transform}_counterexamples.{image_format}')
                    )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir',
        default='./results/gtsb/formal/images',
        help='output directory')
    parser.add_argument('-s', '--sampleindexes',
        nargs='*',
        default=[],
        help='list of indexes and/or ranges of samples to test (e.g. 1 2 10-20 100-110)')
    parser.add_argument('-f', '--formats',
        nargs='*',
        default=['png'],
        help='image format(s) (png and/or pdf)')
    args = parser.parse_args()
    
    main(args.outdir, parse_indexes(args.sampleindexes), args.formats)
