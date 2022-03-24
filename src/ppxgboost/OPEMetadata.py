import re
import numpy as np
import pandas as pd
from ppxgboost.Tree import *
from ppxgboost.Model import *
from ope.pyope.ope import DEFAULT_IN_RANGE_END

# This is the maximum number that the OPE encryption can support.
#   Currently, we also set this to be the maximum number that
#   affine transform map to (see `affine_transform`).
MAX_NUM_OPE_ENC = DEFAULT_IN_RANGE_END

class Metadata:
    """
    This is a metadata structure before encryption. It contains the minimum and maximum value
    from the training dataset as well as the model file

    # Although it would be nice if the model encryption was context-free, the OPE
    # scheme needs to be aware of the min and max values it will need to encrypt.
    # This function returns the min and max values over the test data set to use
    """
    def __init__(self, model, test_data):
        test_data_min = np.min(pd.DataFrame.min(test_data))
        test_data_max = np.max(pd.DataFrame.max(test_data))
        model_min, model_max = model.get_extreme_values()
        self.mini = min(test_data_min, model_min)
        self.maxi = max(test_data_max, model_max)

    # TODO: get rid of this constructor. Currently only used for tests
    # def __init__(self, min_max: dict):
    #     self.mini = min_max['min']
    #     self.maxi = min_max['max']

    def set_min(self, new_min):
        self.mini = new_min

    def set_max(self, new_max):
        self.maxi = new_max

    def affine_transform(self, x):
        """
        This affine transformation will linearly rescale [min, max] to [0, MAX_NUM_AFFINE].
        Linear rescaling:  (x - n_min) * MAX_NUM_AFFINE / (n_max - n_min)
                           MAX_NUM_AFFINE / (n_max - n_min) x - MAX_NUM_AFFINE * n_min
        :param x: input number
        :return: mapping numerical value
        """
        return int((x - self.mini) * MAX_NUM_OPE_ENC / (self.maxi - self.mini))
