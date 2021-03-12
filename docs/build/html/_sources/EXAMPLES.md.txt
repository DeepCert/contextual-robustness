# Examples

Examples showing analysis and reporting for various GTSRB and CIFAR [models](https://github.com/DeepCert/contextual-robustness/tree/main/models) from the DeepCert paper.

## Example Script CLI Arguments

The example scripts support a one or more CLI arguments for things such as specifying the output directory and selecting the sample indexes to test. You can view the CLI options for any of the scripts by passing the `-h` (or `--help`) argument.

## GTSRB Test-Based Example

__Analysis:__

The GTSRB test-based analysis script ([./examples/gtsrb_test_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsrb_test_analysis.py)) analyzes [GTSRB models](https://github.com/DeepCert/contextual-robustness/tree/main/models#gtsrb-models) 1a, 1b, 2a, 2b, 3a, and 3b on all test samples from the GTSRB dataset and saves the results & counterexamples.

```sh
./examples/gtsrb_test_analysis.py
```

__Reporting:__

The GTSRB test-based reporting script ([./examples/gtsrb_test_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsrb_test_reporting.py)) generates epsilon boxplots, counterexample plots, class-accuracy charts, and model-accuracy comparison charts for the results obtained from the gtsrb test-based analysis.

```sh
./examples/gtsrb_test_reporting.py
```

## GTSRB Formal Verification Example

__Analysis:__

The GTSRB formal verification analysis script ([./examples/gtsrb_formal_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsrb_formal_analysis.py)) analyzes [GTSRB models](https://github.com/DeepCert/contextual-robustness/tree/main/models#gtsrb-models) 1a and 1b on 35 test samples from the GTSRB dataset against "haze" and "l-inf" transforms using formal verification and saves the results & counterexamples.

NOTE: If using Gurobi, run analysis with the `-g` option

```sh
./examples/gtsrb_formal_analysis.py -s 19 101 163 350 507 930 1041 1298 1317 2814 1773 1810 1879 2126 2279 2330 2460 2773 2908 2937 453 3309 3339 3470 3538 3560 3609 3700 3750 3798 3828 3875 3942 3948 3984
```

__Reporting:__

The GTSRB formal verification reporting script ([./examples/gtsrb_formal_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/gtsrb_formal_reporting.py)) generates epsilon boxplots and counterexample plots for the results obtained from the GTSRB formal verification analysis.

```sh
./examples/gtsrb_formal_reporting.py -s 19 101 163 350 507 930 1041 1298 1317 2814 1773 1810 1879 2126 2279 2330 2460 2773 2908 2937 453 3309 3339 3470 3538 3560 3609 3700 3750 3798 3828 3875 3942 3948 3984
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
./examples/cifar_test_reporting.py
```

## CIFAR Formal Verification Example

__Analysis:__

The CIFAR formal verification analysis script ([./examples/cifar_formal_analysis.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/cifar_formal_analysis.py)) analyzes [CIFAR models](https://github.com/DeepCert/contextual-robustness/tree/main/models#cifar-models) 4a and 4b on 50 test samples from the CIFAR dataset against the "haze" and "l-inf" transforms using formal verification and saves the results & counterexamples.

NOTE: If using Gurobi, run analysis with the `-g` option

```sh
./examples/cifar_formal_analysis.py -s 64 150 621 698 997 1535 1658 1724 1988 2018 2135 2194 2370 2793 3000 3017 3130 3141 3328 3634 3829 4076 4131 4714 4770 4905 4937 5435 5526 5714 5785 5947 6046 6372 6487 6544 7113 7581 7629 7718 8039 8059 8190 8343 8588 8829 8988 9155 9941 9981
```

__Reporting:__

The CIFAR formal verification reporting script ([./examples/cifar_formal_reporting.py](https://github.com/DeepCert/contextual-robustness/tree/main/examples/cifar_formal_reporting.py)) generates epsilon boxplots and counterexample plots for the results obtained from the CIFAR formal verification analysis.

```sh
./examples/cifar_formal_reporting.py -s 64 150 621 698 997 1535 1658 1724 1988 2018 2135 2194 2370 2793 3000 3017 3130 3141 3328 3634 3829 4076 4131 4714 4770 4905 4937 5435 5526 5714 5785 5947 6046 6372 6487 6544 7113 7581 7629 7718 8039 8059 8190 8343 8588 8829 8988 9155 9941 9981
```
