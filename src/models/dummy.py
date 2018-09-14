import sys
sys.path.insert(0,'..')
import numpy as np

import torch

from conditional_gen import congen_inference
from unconditional_gen import uncongen_inference
import config as args

def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    stroke = uncongen_inference.generate_unconditional(random_seed=random_seed)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer
    stroke = congen_inference.generate_conditional(random_seed=random_seed, text=text)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
