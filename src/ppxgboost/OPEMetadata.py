# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pyope.ope as pyope

class OPEMetadata:
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

    def __init__(self, model, test_data_min, test_data_max, max_num_ope_enc = pyope.DEFAULT_IN_RANGE_END):
        """
        :param model: plaintext PPXGBoost model
        :param test_data_min: minimum value across all features of all (future) queries
        :param test_data_max: maximum value across all features of all (future) queries
        :param max_num_ope_enc: the maximum number that the OPE encryption can support. This value is also used
                                as the maximum output of the affine transformation
        """
        model_min, model_max = model.get_extreme_values()
        self.min_val = min(test_data_min, model_min)
        self.max_val = max(test_data_max, model_max)
        self.max_num_ope_enc = max_num_ope_enc

    def affine_transform(self, x):
        """
        This affine transformation will linearly rescale [min_val, max_val] to [0, max_num_ope_enc].
        Linear rescaling:  (x - min_val) * max_num_ope_enc / (max_val - min_val)

        :param x: float in range [min_val, max_val]
        :return: int in range [0, max_num_ope_enc]
        """
        if x > self.max_val or x < self.min_val:
            raise Exception('Input ' + str(x) + ' is outside allowed range [' +
                            str(self.min_val) + ', ' + str(self.max_val) + ']')

        return int((x - self.min_val) * self.max_num_ope_enc / (self.max_val - self.min_val))
