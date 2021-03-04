#!./venv/bin/python
import os
from contextual_robustness import ContextualRobustnessTest
from contextual_robustness.transforms import test_transforms as transforms
from contextual_robustness.load_data import load_gtsrb
from contextual_robustness.utils import parse_indexes

# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(outdir, sample_indexes):
    # load dataset
    _, _, X_test, Y_test, _ = load_gtsrb()

    # define models
    models = ('1a', '1b', '2a', '2b', '3a', '3b')

    # analyze each model on each transform
    for transform in transforms.keys():
        transform_name = transform.capitalize()
        for m in models:
            model_name = f'Model{m}'
            model_path = f'./models/gtsb/model{m}.h5'
            print(f'{("-"*80)}\nAnalyzing {model_name} {transform_name}\n{("-"*80)}')
            cr = ContextualRobustnessTest(
                model_path=model_path,
                model_name=model_name,
                X=X_test,
                Y=Y_test,
                transform_fn=transforms[transform]['fn'],
                transform_args=transforms[transform]['args'],
                transform_name=transform_name,
                sample_indexes=sample_indexes
                )
            cr.analyze(
                epsilons_outpath=os.path.join(outdir, f'model{m}-{transform}.csv'),
                counterexamples_outpath=os.path.join(outdir, f'model{m}-{transform}-counterexamples.p')
                )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir',
        default='./results/gtsb/test/data',
        help='output directory')
    parser.add_argument('-s', '--sampleindexes',
        nargs='*',
        default=[],
        help='list of indexes and/or ranges of samples to test (e.g. 1 2 10-20 100-110)')
    args = parser.parse_args()
    
    main(args.outdir, parse_indexes(args.sampleindexes))
