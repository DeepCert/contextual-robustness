# Examples

Examples showing analysis and reporting for various GTSRB and CIFAR [models](https://github.com/DeepCert/contextual-robustness/tree/main/models) from the DeepCert paper.

## Example Script CLI Arguments

The example scripts support a one or more CLI arguments for things such as specifying the output directory and selecting the sample indexes to test. You can view the CLI options for any of the scripts by passing the `-h` (or `--help`) argument.

## GTSRB Test-Based Example

__Analysis:__

The GTSRB test-based analysis script ([./examples/gtsb_test_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsb_test_analysis.py)) analyzes [GTSRB models](https://github.com/DeepCert/contextual-robustness/tree/main/models#gtsrb-models) 1a, 1b, 2a, 2b, 3a, and 3b on all test samples from the GTSRB dataset and saves the results & counterexamples.

```sh
./examples/gtsb_test_analysis.py
```

__Reporting:__

The GTSRB test-based reporting script ([./examples/gtsb_test_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsb_test_reporting.py)) generates epsilon boxplots, counterexample plots, class-accuracy charts, and model-accuracy comparison charts for the results obtained from the gtsrb test-based analysis.

```sh
./examples/gtsb_test_reporting.py
```

## GTSRB Formal Verification Example

__Analysis:__

The GTSRB formal verification analysis script ([./examples/gtsb_formal_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsb_formal_analysis.py)) analyzes [GTSRB models](https://github.com/DeepCert/contextual-robustness/tree/main/models#gtsrb-models) 1a and 1b on 35 test samples from the GTSRB dataset against "haze" and "l-inf" transforms using formal verification and saves the results & counterexamples.

```sh
./examples/gtsb_formal_analysis.py -s 8 38-39 47 67 68 150 443 480 508 542 810 853 954 1117 1376 1483 1511 1948 2026 2227 2396 2559 2609 2731 2882 3005 3230 3344 3548 3749 3825 4033 4076 4090
```

__Reporting:__

The GTSRB formal verification reporting script ([./examples/gtsb_formal_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsb_formal_reporting.py)) generates epsilon boxplots and counterexample plots for the results obtained from the GTSRB formal verification analysis.

```sh
./examples/gtsb_formal_reporting.py -s 8 38-39 47 67 68 150 443 480 508 542 810 853 954 1117 1376 1483 1511 1948 2026 2227 2396 2559 2609 2731 2882 3005 3230 3344 3548 3749 3825 4033 4076 4090
```

## CIFAR Test-Based Example

__Analysis:__

The CIFAR test-based analysis script ([./examples/cifar_test_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/cifar_test_analysis.py)) analyzes [CIFAR models](https://github.com/DeepCert/contextual-robustness/tree/main/models#cifar-models) 4a, 4b, 5a, 5b, 6a, and 6b on all test samples from the CIFAR dataset against the "haze", "contrast", and "blur" transforms using the test-based technique and saves the results & counterexamples.

```sh
./examples/cifar_test_analysis.py
```

__Reporting:__

The CIFAR test-based reporting script ([./examples/cifar_test_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/cifar_test_reporting.py)) generates epsilon boxplots, counterexample plots, class-accuracy charts, and model-accuracy comparison charts for the results obtained from the CIFAR test-based analysis.

```sh
./examples/gtsb_test_reporting.py
```

## CIFAR Formal Verification Example

__Analysis:__

The CIFAR formal verification analysis script ([./examples/cifar_formal_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/cifar_formal_analysis.py)) analyzes [CIFAR models](https://github.com/DeepCert/contextual-robustness/tree/main/models#cifar-models) 4a and 4b on 50 test samples from the CIFAR dataset against the "haze" and "l-inf" transforms using formal verification and saves the results & counterexamples.

```sh
./examples/cifar_formal_analysis.py -s 9 37 103 183 255 372 398 684 845 966 1196 1200 1878 1988 2081 2135 2246 2372 2584 2663 2954 3004 3497 3510 3858 4195 5047 5087 5195 5587 5616 5644 5858 6518 6622 6915 8213 8562 9090 9187 9198 9261 9358 9472 9605 9723 9749 9810 9941 9974
```

__Reporting:__

The CIFAR formal verification reporting script ([./examples/cifar_formal_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/cifar_formal_reporting.py)) generates epsilon boxplots and counterexample plots for the results obtained from the CIFAR formal verification analysis.

```sh
./examples/cifar_formal_reporting.py -s 9 37 103 183 255 372 398 684 845 966 1196 1200 1878 1988 2081 2135 2246 2372 2584 2663 2954 3004 3497 3510 3858 4195 5047 5087 5195 5587 5616 5644 5858 6518 6622 6915 8213 8562 9090 9187 9198 9261 9358 9472 9605 9723 9749 9810 9941 9974
```
