#!./venv/bin/python
import os
from contextual_robustness import ContextualRobustnessFormal
from contextual_robustness.transforms import formal_transforms as transforms
from contextual_robustness.datasets import load_gtsrb
from contextual_robustness.utils import remove_softmax_activation, parse_indexes

# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(models, transform_names, outdir, sample_indexes):
    # load dataset
    _, _, X_test, Y_test, _ = load_gtsrb()

    # analyze each model on each transform
    for transform in transform_names:
        transform_name = transform.capitalize()
        for m in models:
            model_name = f'Model{m}'
            print(f'{("-"*80)}\nAnalyzing {model_name} {transform_name}\n{("-"*80)}')
            
            # create a copy of the h5 model without softmax activation
            model_path = f'./models/gtsrb/model{m}-verification'
            remove_softmax_activation(f'./models/gtsrb/model{m}.h5', save_path=model_path)
            
            # run analysis on modified model
            cr = ContextualRobustnessFormal(
                model_path=model_path,
                model_name=model_name,
                model_args=dict(modelType='savedModel_v2'),
                transform_fn=transforms[transform]['fn'],
                transform_args=transforms[transform]['args'],
                transform_name=transform_name,
                X=X_test,
                Y=Y_test,
                sample_indexes=sample_indexes,
                verbosity=1
                )
            cr.analyze(
                epsilons_outpath=os.path.join(outdir, f'model{m}-{transform}.csv'),
                counterexamples_outpath=os.path.join(outdir, f'model{m}-{transform}-counterexamples.p')
                )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--models',
        nargs='+',
        default=['1a', '1b'],
        help='model(s) to analyze')
    parser.add_argument('-t', '--transforms',
        nargs='+',
        default=['encode_haze', 'encode_linf'],
        help='image transform(s) to test')
    parser.add_argument('-o', '--outdir',
        default='./results/gtsrb/formal/data',
        help='output directory')
    parser.add_argument('-s', '--sampleindexes',
        nargs='*',
        default=[],
        help='list of indexes and/or ranges of samples to test (e.g. 1 2 10-20 100-110)')
    args = parser.parse_args()
    
    main(args.models, args.transforms, args.outdir, parse_indexes(args.sampleindexes))
