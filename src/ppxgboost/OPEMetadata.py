import re
import numpy as np
import pandas as pd
from ppxgboost.Tree import *
from ppxgboost.Model import *
from ope.pyope.ope import DEFAULT_IN_RANGE_END

# This is the maximum value that the OPE encryption can support.
# `affine_transform` uses this value as the upper bound for the
# OPE range.
MAX_NUM_OPE_ENC = DEFAULT_IN_RANGE_END

class Metadata:
    """
    OPE encryption takes floating point values in a fixed range and
    maps them to an integer range. This metadata defines the input
    range for this OPE instance.

    Although it would be nice if the model encryption was context-free,
    evaluating a model requires comparing values in an (unknown at encryption time)
    query to values in the model itself. Thus the input range for the OPE
    instance depends on the comparison values in the model itself as well
    as the minimum and maximum values in any query, across all features.
    """
    def __init__(self, model, test_data_min, test_data_max):
        model_min, model_max = model.get_extreme_values()
        self.min_val = min(test_data_min, model_min)
        self.max_val = max(test_data_max, model_max)

    def affine_transform(self, x):
        """
        This affine transformation will linearly rescale [min, max] to [0, MAX_NUM_AFFINE].
        Linear rescaling:  (x - n_min) * MAX_NUM_AFFINE / (n_max - n_min)
                           MAX_NUM_AFFINE / (n_max - n_min) x - MAX_NUM_AFFINE * n_min
        :param x: input number
        :return: mapping numerical value
        """
        return int((x - self.min_val) * MAX_NUM_OPE_ENC / (self.max_val - self.min_val))

# For tests and examples, the entire test dataset is known in advance
# so we can just extract the minimum and maximum values from the dataset
# rather than guessing.
def test_data_extreme_values(test_data):

    data_min = float('inf')
    data_max = float('-inf')
    for x in test_data:
        values = x.values()
        q_min = min(values)
        q_max = max(values)
        data_min = min(q_min, data_min)
        data_max = max(q_max, data_max)

    return data_min, data_max
