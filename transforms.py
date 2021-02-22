
import numpy as np
import scipy.stats as st
from scipy.signal import convolve2d

'''
Test Verification Transform Functions
-------------------------------------
Applied to an image; Used by test-based verification technique.

Arguments:
    network (MarabouNetwork) - the MarabouNetwork object
    image   (np.array)       - the image to transform
    epsilon (float)          - amount of transform to apply to image
    output  (integer)        - output index to solve for
    ...                      - optional keyword args passed by 'transform_args'

Returns:
    (MarabouNetwork) - the network with encoded image transformation
'''

def haze(image, epsilon):
    fog = np.ones_like(image)
    fog[:, :, 0] *= 1.0  # red
    fog[:, :, 1] *= 1.0  # green
    fog[:, :, 2] *= 1.0  # blue
    return (1-epsilon) * image[:, :, :] + epsilon * fog[:, :, :]

def increaseContrast(image, epsilon):
    # This is the scaling we would prefer
    tg_min = 0.0
    tg_max = 1.0

    # This is the max and minimum value in the picture originally
    sc_min = 0.5*epsilon
    sc_max = 1 - sc_min
    output = (image - sc_min) * (tg_max - tg_min) / (sc_max - sc_min) + tg_min
    return np.clip(output, 0, 1)

def gaussianblureps(imageIn, epsilon, kernelSize=17, scaling=20):
    image = imageIn.copy()
    nsig = (0.01-scaling)*epsilon + scaling
    x = np.linspace(-nsig, nsig, kernelSize+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    kernel =  kern2d/kern2d.sum()

    for i in range(3):
        image[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same', boundary='symm')
    return image

test_transforms = {
    'haze': {'fn': haze, 'args': dict()},
    'contrast': {'fn': increaseContrast, 'args': dict()},
    'blur': {'fn': gaussianblureps, 'args': dict(kernelSize=17, scaling=20)}
    }

'''
Formal Verification Transform Encoder Functions
-----------------------------------------------
Used by formal verification technique; Encodes a query for an image transformation 
as a Marabou input query;

Arguments:
    network       (MarabouNetwork) - the MarabouNetwork object
    image         (np.array)       - the image to transform
    epsilon       (float)          - amount of transform to apply to image
    output_index  (integer)        - output index to solve for
    ...                            - optional arguments passed by 'transform_args'

Returns:
    (MarabouNetwork) - the network with encoded image transformation
'''

def encode_haze(self, network, image, epsilon, output_index):
    n_inputs = network.inputVars[0].flatten().shape[0]
    n_outputs = network.outputVars[0].flatten().shape[0]
    flattened_image = image.flatten()
    eps = network.getNewVariable()
    network.setLowerBound( eps, 0 )
    network.setUpperBound( eps, epsilon )
    network.inputVars = np.array([eps])
    for i in range(n_inputs):
        val = flattened_image[i]
        network.addEquality([i, eps], [1, val - 1], val)
    for i in range(n_outputs):
        if i != output_index:
            network.addInequality([network.outputVars[0][i], network.outputVars[0][output_index]], [1, - 1], 0)
    return network,eps

def encode_linf(self, network, image, epsilon, output_index):
    n_inputs = network.inputVars[0].flatten().shape[0]
    n_outputs = network.outputVars[0].flatten().shape[0]
    flattened_image = image.flatten()
    for i in range(n_inputs):
        val = flattened_image[i]
        network.setLowerBound(i, max(0, val - epsilon))
        network.setUpperBound(i, min(1, val + epsilon))
    for i in range(n_outputs):
        if i != output_index:
            network.addInequality([network.outputVars[0][i], network.outputVars[0][output_index]], [1, - 1], 0)
    return network

formal_transforms = {
    'encode_haze': {'fn': encode_haze, 'args': dict()},
    'encode_linf': {'fn': encode_linf, 'args': dict()}
    }
