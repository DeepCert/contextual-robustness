# Contextual-Robustness

Contextual robustness verification source code

## Install & Setup

```sh
git clone <GITHUB_REPO_URL>
cd contextual-robustness

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

./scripts/setup.sh
```

By default, the plotting functions in `ContextualRobustnessReporting` use [LaTeX](https://www.latex-project.org/get/) for rendering text. To use this feature, you'll need to [install LaTeX](https://www.latex-project.org/get/) and add it to your `PATH`. If you don't need/want to use the latex rendering, simply pass `usetex=False` to the relevant plotting functions.

## Usage Examples

### Test-Based Technique

The test based technique uses the model's built-in 'predict' function to analyze the model's robustness to a transform. This technique can be used to get an overall picture of the model's robustness by testing a large number of samples from the dataset. It is faster than the formal verification technique, but may not always discover the real "epsilon" for all samples.

#### Test-Based Technique: Analyzing a Model/Transform

```python
from contextual_robustness import ContextualRobustnessTest, ContextualRobustnessReporting
from transforms import haze

# instantiate ContextualRobustness object for modelA/haze
modelA_haze_test = ContextualRobustnessTest(
    model_name='./models/modelA.h5', # (*required) model
    model_name='ModelA',             # name of model
    transform_fn=haze,               # (*required) transform function
    transform_name='Haze',           # name of transform
    X=X_test,                        # (*required) np.array of images
    Y=Y_test,                        # (*required) np.array of labels
    verbosity=0                      # amount of logging
    )
# run analysis and save to CSV
modelA_haze_test.analyze(
    epsilons_outpath='./results/modelA_haze/epsilons.csv',
    counterexamples_outpath='./results/modelA_haze/counterexamples.p'
    )
```

#### Test-Based Technique: Load & Visualize Results

```python
from contextual_robustness import ContextualRobustnessTest, ContextualRobustnessReporting
from transforms import haze

# Load saved CSV results from 'Haze' analysis on 'ModelA'
modelA_haze = ContextualRobustnessTest(
    model_path='./models/modelA.h5',
    model_name='ModelA',
    X=X,
    Y=Y,
    transform_fn=haze,
    transform_name='Haze'
    )
modelA_haze.load_results(
    epsilons_path='./results/modelA_haze/epsilons.csv',
    counterexamples_path='./results/modelA_haze/counterexamples.p'
    )
# Load saved CSV results from 'Haze' analysis on 'ModelB'
modelB_haze = ContextualRobustnessTest(
    model_path='./models/modelB.h5',
    model_name='ModelB',
    X=X,
    Y=Y,
    transform_fn=haze,
    transform_name='Haze'
    )
modelB_haze.load_results(
    epsilons_path='./results/modelB_haze/epsilons.csv',
    counterexamples_path='./results/modelB_haze/counterexamples.p'
    )

# Generate individual 'epsilon' boxplots for each model
ContextualRobustnessReporting.generate_epsilons_plot(
    modelA_haze,
    outfile='./results/modelA_haze/epsilons.png')
ContextualRobustnessReporting.generate_epsilons_plot(
    modelB_haze,
    outfile='./results/modelB_haze/epsilons.png')

# Generate individual 'counterexample' plots for each model
ContextualRobustnessReporting.generate_counterexamples_plot(
    modelA_haze,
    outfile='./results/modelA_haze/counterexamples.png')
ContextualRobustnessReporting.generate_counterexamples_plot(
    modelB_haze,
    outfile='./results/modelB_haze/counterexamples.png')

# Generate individual 'class accuracy' line charts for each model
ContextualRobustnessReporting.generate_class_accuracy_plot(
    modelA_haze,
    outfile='./results/modelA_haze/class-accuracy.png')
ContextualRobustnessReporting.generate_class_accuracy_plot(
    modelB_haze,
    outfile='./results/modelB_haze/class-accuracy.png')

# Generate haze accuracy report comparing the transform on multiple models
ContextualRobustnessReporting.generate_accuracy_report_plot(
    cr_objects=[modelA_haze, modelB_haze],
    outfile='./results/haze-accuracy-report.png')
```

### Formal Verification Technique

The formal verification technique uses the [Marabou neural network verification framework](https://github.com/NeuralNetworkVerification/Marabou) to evaluate the model's robustness to a transform encoded as a Marabou input query. This method provides the true "epsilon" and formal guarantees about the model's robustness to the transform, however it takes significantly longer than the test-based method. Using the `sample_indexes` option to select a subset of samples to test is recommended.

#### Formal Verification Technique: Model Preparation

Marabou relies on the output of the network's logits layer, so if the network has a softmax output layer, the activation function will need to be removed from that layer. The `remove_softmax_activation` function is supplied to do this for Tensorflow v2 models. The example below shows how to use `remove_softmax_activation` to save a copy of the model without the softmax activation function on the output layer.

```python
from utils import remove_softmax_activation

# save a copy the model without softmax activation function
remove_softmax_activation('./modelA.h5', save_path='./modelA-verification')
```

#### Formal Verification Technique: Analyzing a Model/Transform

```python
import sys
from contextual_robustness import ContextualRobustnessFormal, ContextualRobustnessReporting
from transforms import encode_haze

sys.path.append('../Marabou/')
from maraboupy import Marabou

# Instantiate ContextualRobustness object for modelA/haze
modelA_haze_formal = ContextualRobustnessFormal(
    model_path='modelA-verification',           # (*required) path to model
    model_name='ModelA',                        # name of model
    model_args=dict(modelType='savedModel_v2'), # specify model type for marabou
    transform_fn=encode_haze,                   # (*required) transform encoder function
    transform_name='Haze',                      # name of transform
    X=X,                                        # (*required) np.array of images
    Y=Y,                                        # (*required) np.array of labels
    sample_indexes=list(range(0,10)),           # list of indexes of samples to test
    verbosity=0                                 # amount of logging
    )
# run analysis and save to CSV
modelA_haze_formal.analyze(
    epsilons_outpath='./results/modelA_haze_formal/epsilons.csv',
    counterexamples_outpath='./results/modelA_haze_formal/counterexamples.p'
    )
```

#### Formal Verification Technique: Load & Visualize Results

```python
from contextual_robustness import ContextualRobustnessFormal, ContextualRobustnessReporting
from transforms import encode_haze

sys.path.append('../Marabou/')
from maraboupy import Marabou

# Load saved CSV results from 'Haze' analysis on 'ModelA'
modelA_haze_formal = ContextualRobustnessFormal(
    model_path='modelA-verification',           # (*required) path to model
    model_name='ModelA',                        # name of model
    model_args=dict(modelType='savedModel_v2'), # specify model type for marabou
    transform_fn=encode_haze,                   # (*required) transform encoder function
    transform_name='Haze',                      # name of transform
    X=X,                                        # (*required) np.array of images
    Y=Y,                                        # (*required) np.array of labels
    sample_indexes=list(range(0,10)),           # list of indexes of samples to test
    verbosity=0                                 # amount of logging
    )
modelA_haze_formal.load_results(
    epsilons_path='./results/modelA_haze/epsilons.csv',
    counterexamples_path='./results/modelA_haze_formal/counterexamples.p'
    )

# Generate individual 'epsilon' boxplots for each model
ContextualRobustnessReporting.generate_epsilons_plot(
    modelA_haze_formal,
    outfile='./results/modelA_haze_formal/epsilons.png'
    )

# Generate individual 'counterexample' plots for each model
ContextualRobustnessReporting.generate_counterexamples_plot(
    modelA_haze_formal,
    outfile='./results/modelA_haze_formal/counterexamples.png'
    )
```

## Full Examples

Example code used to analyze and generate reports for the GTSB models (1a, 1b, 2a, 2b, 3a, 3b) and CIFAR models (4a, 4b, 5a, 5b, 6a, 6b) from the DeepCert paper. The models can be found in the [./models](./models) folder, and the code can be found in the [./examples](./examples) folder.

* Analyze models 4a-6b on haze, blur, and contrast with test-based technique: [./examples/cifar_test_analysis.py](./examples/cifar_test_analysis.py)
* Generate plots from analysis of models 4a-6b on haze, blur, and contrast: [./examples/cifar_test_reporting.py](./examples/cifar_test_analysis.py)
* Analyze models 4a-6b on haze, blur, and contrast with formal verification technique: [./examples/cifar_formal_analysis.py](./examples/cifar_formal_analysis.py)

## Resources

* [Marabou Documentation](https://neuralnetworkverification.github.io/Marabou/)
