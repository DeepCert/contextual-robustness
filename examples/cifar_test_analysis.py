#!/usr/bin/python3
import os
from contextual_robustness import ContextualRobustnessTest
from transforms import test_transforms as transforms
from load_data import load_cifar

# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(outdir):
    # load dataset
    _, _, X_test, Y_test, _ = load_cifar()

    # define transforms & models
    models = ('4a', '4b', '5a', '5b', '6a', '6b')

    # analyze each model on each transform
    for transform in transforms.keys():
        transform_name = transform.capitalize()
        for m in models:
            model_name = f'Model{m}'
            model_path = f'./models/cifar/model{m}.h5'
            print(f'{("-"*80)}\nAnalyzing {model_name} {transform_name}\n{("-"*80)}')
            cr = ContextualRobustnessTest(
                model_path=model_path,
                model_name=model_name,
                X=X_test,
                Y=Y_test,
                transform_fn=transforms[transform]['fn'],
                transform_args=transforms[transform]['args'],
                transform_name=transform_name
                )
            cr.analyze(
                epsilons_outpath=os.path.join(outdir, f'model{m}-{transform}.csv'),
                counterexamples_outpath=os.path.join(outdir, f'model{m}-{transform}-counterexamples.p')
                )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir',
        default='./results/cifar/test/data',
        help='output directory')
    args = parser.parse_args()
    
    main(args.outdir)
