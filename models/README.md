# SafeComp2021 CIFAR Models

## Models 7 & 7a

A simple two-layer fully-connected, feed-forward ReLU model trained on CIFAR-10. Accuracy varies between 47% and 50%.

```text
Input   : (32,32,3)
Flatten : (32,32,3) -> (3072,)
Layer1  : 32 nodes, ReLU
Layer2  : 16 nodes, ReLU
Output  : (10,) Softmax
```

## Models 8 & 8a

A simple four-layer, feed-forward, fully-connected ReLU model trained on CIFAR-10. Accuracy varies between 52% and 54%.

```text
Input   : (32,32,3)
Flatten : (32,32,3) -> (3072,)
Layer1  : 128 nodes, ReLU
Layer2  : 64 nodes, ReLU
Layer3  : 32 nodes, ReLU
Layer4  : 16 nodes, ReLU
Output  : (10,) Softmax
```

## Models 9 & 9a

CNNs trained on CIFAR-10 consisting of six convolutional ReLU layers, one fully-connected layer, and softmax output. Two additional convolutional layers with no activation and stride > 1 are used to replace maxpooling as described in "Striving for Simplicity: The All Convolutional Net". Accuracy is between 84% and 86%.

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
