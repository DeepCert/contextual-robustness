# contextual-robustness

Contextual robustness verification source code

## Install & Setup

```sh
git clone <GITHUB_REPO_URL>
cd contextual-robustness

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

./scripts/setup.sh
```

By default, the plotting functions in `ContextualRobustnessReporting` use [LaTeX](https://www.latex-project.org/get/) for rendering text. To use this feature, you'll need to [install LaTeX](https://www.latex-project.org/get/) and add it to your `PATH`. If you don't need/want to use the latex rendering, simply pass `usetex=False` to the relevant plotting functions.

## Usage Examples

### Test Based Technique

The test based technique uses the model's built-in 'predict' function to analyze the model's robustness to a transform.

#### Analyzing Model/Transform Using Test Technique

```python
import tensorflow as tf
from contextual_robustness import ContextualRobustnessTest, ContextualRobustnessReporting
from transforms import haze

# Load model as a MarabouNetwork object
model = tf.keras.models.load_model('./models/model1.h5')

# instantiate ContextualRobustness object for model1/haze
model1_haze = ContextualRobustnessTest(
    model=model,              # (*required) model
    model_name='Model1',      # name of model
    transform_fn=encode_haze, # (*required) transform function
    transform_name='Haze',    # name of transform
    X=X_test,                 # (*required) np.array of images
    Y=Y_test)                 # (*required) np.array of labels

# run analysis and save to CSV
model1_haze.analyze(
    outfile='./results/model1_haze/epsilons.csv',
    verbose=0)
```

### Formal Verification Technique

The formal verification technique uses the [Marabou neural network verification framework](https://github.com/NeuralNetworkVerification/Marabou) to evaluate the model's robustness to a transform encoded as a Marabou input query.

#### Model Preparation

Marabou relies on the output of the network's logits layer, so if the network has a softmax output layer, the activation function will need to be removed from that layer. The `prepare_classifier` function is supplied to do this for Tensorflow v2 models. The example below shows how to use `prepare_classifier` to save a copy of the model without the softmax activation function on the output layer.

```python
from utils import prepare_classifier

model = prepare_classifier('./modelX.h5')
model.save('./modelX-verification')
```

#### Analyzing Model/Transform Using Formal Technique

```python
import sys
from contextual_robustness import ContextualRobustnessFormal, ContextualRobustnessReporting
from transforms import encode_haze
sys.path.append('../Marabou/')
from maraboupy import Marabou

# Load model as a MarabouNetwork object
model = Marabou.read_tf('./models/model1-verification', modelType='savedModel_v2')

# Formal verification is CPU intensive, so choose a subset of samples
X, Y = X_test[0:10], Y_test[0:10]

# instantiate ContextualRobustness object for model1/haze
model1_haze_formal = ContextualRobustnessFormal(
    model=model,              # (*required) MarabouNetwork model
    model_name='Model1',      # name of model
    transform_fn=encode_haze, # (*required) transform encoder function
    transform_name='Haze',    # name of transform
    X=X,                      # (*required) np.array of images
    Y=Y)                      # (*required) np.array of labels

# run analysis and save to CSV
model1_haze_formal.analyze(
    outfile='./results/model1_haze_formal/epsilons.csv',
    verbose=0)
```

### Load & Visualize Results

```python
from contextual_robustness import ContextualRobustnessTest, ContextualRobustnessReporting
from transforms import haze

# Load saved CSV results from 'Haze' analysis for two different models
model1_haze = ContextualRobustnessTest(
    model_path='./models/model1.h5',
    model_name='Model1',
    X=X,
    Y=Y,
    transform_fn=haze,
    transform_name='Haze'
    ).load_results('./results/model1_haze/epsilons.csv')
model1_haze = ContextualRobustnessTest(
    model_path='./models/model2.h5',
    model_name='Model2',
    X=X,
    Y=Y,
    transform_fn=haze,
    transform_name='Haze'
    ).load_results('./results/model2_haze/epsilons.csv')

# Generate individual 'epsilon' boxplots for each model
ContextualRobustnessReporting.generate_epsilons_plot(
    model1_haze,
    outfile='./results/model1_haze/epsilons.png')
ContextualRobustnessReporting.generate_epsilons_plot(
    model2_haze,
    outfile='./results/model2_haze/epsilons.png')

# Generate individual 'counterexample' plots for each model
ContextualRobustnessReporting.generate_counterexamples_plot(
    model1_haze,
    outfile='./results/model1_haze/counterexamples.png')
ContextualRobustnessReporting.generate_counterexamples_plot(
    model2_haze,
    outfile='./results/model2_haze/counterexamples.png')

# Generate individual 'class accuracy' line charts for each model
ContextualRobustnessReporting.generate_class_accuracy_plot(
    model1_haze,
    outfile='./results/model1_haze/class-accuracy.png')
ContextualRobustnessReporting.generate_class_accuracy_plot(
    model2_haze,
    outfile='./results/model2_haze/class-accuracy.png')

# Generate haze accuracy report comparing the transform on multiple models
ContextualRobustnessReporting.generate_accuracy_report_plot(
    cr_objects=[model1_haze, model2_haze],
    outfile='./results/haze-accuracy-report.png'
    )
```

## Resources

* [Marabou Documentation](https://neuralnetworkverification.github.io/Marabou/)
