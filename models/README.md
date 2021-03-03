# DeepCert Example Models

## GTSRB Models

Models trained on the [German Traffic Signs Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html) (GTSRB) dataset.

## Models 1a & 1b

A two-layer fully-connected feed-forward ReLU model trained on 7 classes of the GTSRB dataset. Accuracy varies between 81% and 85%.

```text
Input   : (32,32,3)
Flatten : (32,32,3) -> (3072,)
Layer1  : 50 nodes, ReLU
Layer2  : 20 nodes, ReLU
Output  : (7,) Softmax
```

## Models 2a & 2b

A three-layer fully-connected feed-forward ReLU model trained on 7 classes of the GTSRB dataset. Accuracy varies between 86% and 87%.

```text
Input   : (32,32,3)
Flatten : (32,32,3) -> (3072,)
Layer1  : 200 nodes, ReLU
Layer2  : 200 nodes, ReLU
Layer3  : 200 nodes, ReLU
Output  : (7,) Softmax
```

## Models 3a & 3b

A CNN trained on 7 classes of the GTSRB dataset consisting of four convolutional layers with ReLU activation, two max pooling layers, and one fully-connected layer. Accuracy varies between 98% and 99%.

```text
Input   : (32,32,3)
 
Layer1  : Conv2D 32, ReLU
Layer2  : Conv2D 32, ReLU
Layer3  : MaxPooling 2,2
Layer4  : Dropout (0.25)
 
Layer5  : Conv2D 64, ReLU
Layer6  : Conv2D 64, ReLU
Layer7  : MaxPooling 2,2
Layer8  : Dropout (0.25)
 
Flatten : (32,32,3) -> (3072,)
Layer9 : 512 nodes, ReLU
Layer10 : Dropout (0.5)
 
Output  : (7,) Softmax
```

## CIFAR Models

Models trained on the [Canadian Institute For Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html) (CIFAR-10) dataset.

### Models 4a & 4b

A two-layer fully-connected, feed-forward ReLU model trained on CIFAR-10. Accuracy varies between 47% and 50%.

```text
Input   : (32,32,3)
Flatten : (32,32,3) -> (3072,)
Layer1  : 32 nodes, ReLU
Layer2  : 16 nodes, ReLU
Output  : (10,) Softmax
```

### Models 5a & 5b

A four-layer, feed-forward, fully-connected ReLU model trained on CIFAR-10. Accuracy varies between 52% and 54%.

```text
Input   : (32,32,3)
Flatten : (32,32,3) -> (3072,)
Layer1  : 128 nodes, ReLU
Layer2  : 64 nodes, ReLU
Layer3  : 32 nodes, ReLU
Layer4  : 16 nodes, ReLU
Output  : (10,) Softmax
```

### Models 6a & 6b

CNNs trained on CIFAR-10 consisting of six convolutional ReLU layers, one fully-connected layer, and softmax output. Two additional convolutional layers with no activation and stride > 1 are used to replace maxpooling as described in [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806v1). Accuracy is between 84% and 86%.

```text
Input   : (32,32,3)
 
Layer1  : Conv2D 32, ReLU
Layer2  : Dropout (0.2)
Layer3  : Conv2D 32, ReLU
Layer4  : Conv2D 32, strides=2,2 (replaces maxpooling)
 
Layer5  : Conv2D 64, ReLU
Layer6  : Dropout (0.2)
Layer7  : Conv2D 64, ReLU
Layer8  : Conv2D 64, strides=2,2 (replaces maxpooling)
 
Layer9  : Conv2D 128, ReLU
Layer10 : Dropout (0.2)
Layer11 : Conv2D 128, ReLU
Layer12 : Conv2D 128, strides=2,2 (replaces maxpooling)
 
Flatten : (32,32,3) -> (3072,)
Layer10 : 1024 nodes, ReLU
Layer11 : Dropout (0.2)
Layer12 : 512 nodes, ReLU
Layer13 : Dropout (0.2)
 
Output  : (10,) Softmax
```
