import numpy as np
from torch import from_numpy

def make_channel_first(image):
    return np.rollaxis(image, 2, 0)

def make_channel_last(image):
    return np.rollaxis(image, 0, 3)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def to_tensor(array, device):
    if type(array) is tuple:
        return [from_numpy(item).float().to(device) for item in array]
    return from_numpy(array).float().to(device)

def generate_action_space(act_dim, low, high, steps=4):
    """
    IN:
    - steps: steps in each direction from 0
    - act_dim: count of number of actions
    - low: maximum action magnitude
    - high: minimum action magnitude
    """
    high = int(high)
    actions = [tuple([0]*act_dim)] # add no move action
    for i in range(steps):
        for j in np.linspace(high/steps,high,steps):
            a = np.eye(act_dim)[i]*j
            actions.append(tuple(a))
    for i in range(steps):
        for j in np.linspace(high/steps,high,steps):
            a = np.eye(act_dim)[i]*(-j)
            actions.append(tuple(a))
    return actions