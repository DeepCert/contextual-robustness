
import sys
import numpy as np
import scipy.stats as st
from scipy.signal import convolve2d

sys.path.append('./marabou')
from maraboupy import MarabouNetwork

# ----------------------------------------------------------------------------------------------
# Test Verification Transform Functions
# ----------------------------------------------------------------------------------------------
# Apply transform to an image; Used by test-based verification technique.
#
# Args:
#     image   (np.array)       - the image to transform
#     epsilon (float)          - amount of transform to apply to image
#     ...                      - optional keyword args passed by 'transform_args'
#
# Returns:
#     (np.array) - the transformed image
# ----------------------------------------------------------------------------------------------

def haze(image:np.array, epsilon:float) -> np.array:
    '''Applies haze transform to an image

    Args:
        image (np.array): The input image
        epsilon (float): amount of transform

    Returns:
        [np.array]: image with haze
    '''    
    fog = np.ones_like(image)
    fog[:, :, 0] *= 1.0  # red
    fog[:, :, 1] *= 1.0  # green
    fog[:, :, 2] *= 1.0  # blue
    return (1-epsilon) * image[:, :, :] + epsilon * fog[:, :, :]

def increase_contrast(image:np.array, epsilon:float, tg_min:float=0.0, tg_max:float=1.0) -> np.array:
    '''Increases the contrast of the input image

    Args:
        image (np.array): The input image
        epsilon (float): Amount of transform
        tg_min (float, optional): Min value of image scaling. Defaults to 0.0.
        tg_max (float, optional): Max value of image scaling. Defaults to 1.0.

    Returns:
        np.array: The transformed image
    '''
    # this is a hack to prevent div by zero
    if epsilon >= 1.0:
        epsilon = 0.99999
    # This is the max and minimum value in the picture originally
    sc_min = 0.5*epsilon
    sc_max = 1 - sc_min
    output = (image - sc_min) * (tg_max - tg_min) / (sc_max - sc_min) + tg_min
    return np.clip(output, 0, 1)

def gaussianblureps(image:np.array, epsilon:float, kernelSize:int=17, scaling:int=20) -> np.array:
    '''Applies gaussian blur transform to input image

    Args:
        image (np.array): The input image
        epsilon (float): Amount of transform
        kernelSize (int, optional): The kernel size. Defaults to 17.
        scaling (int, optional): Scaling of the blur transform. Defaults to 20.

    Returns:
        np.array: The transformed image
    '''    
    image = image.copy()
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
    'contrast': {'fn': increase_contrast, 'args': dict()},
    'blur': {'fn': gaussianblureps, 'args': dict(kernelSize=17, scaling=20)}
    }

# ----------------------------------------------------------------------------------------------
# Formal Verification Transform Encoder Functions
# ----------------------------------------------------------------------------------------------
# Used by formal verification technique; Encodes a query for an image transformation 
# as a Marabou input query;
#
# Args:
#     network       (MarabouNetwork) - the MarabouNetwork object
#     image         (np.array)       - the image to transform
#     epsilon       (float)          - amount of transform to apply to image
#     output_index  (integer)        - output index to solve for
#     ...                            - optional arguments passed by 'transform_args'
#
# Returns:
#     (MarabouNetwork) - the network with encoded image transformation
# ----------------------------------------------------------------------------------------------

def encode_haze(network:MarabouNetwork, image:np.array, epsilon:float, output_index:int) -> MarabouNetwork:
    '''Encodes a haze transformation as a Marabou input query

    Args:
        network (MarabouNetwork): the MarabouNetwork object
        image (np.array): The input image
        epsilon (float): Amount of transform
        output_index (int): Target output node (for the expected class)

    Returns:
        MarabouNetwork: The MarabouNetwork object with the encoded input query
    '''
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
    return network

def encode_linf(network:MarabouNetwork, image:np.array, epsilon:float, output_index:int) -> MarabouNetwork:
    '''Encodes a linear perturbation as a Marabou input query

    Args:
        network (MarabouNetwork): the MarabouNetwork object
        image (np.array): The input image
        epsilon (float): Amount of transform
        output_index (int): Target output node (for the expected class)

    Returns:
        MarabouNetwork: The MarabouNetwork object with the encoded input query
    '''
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
