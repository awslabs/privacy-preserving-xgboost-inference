# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ope.pyope.ope import DEFAULT_IN_RANGE_END

# This is the maximum value that the OPE encryption can support.
# `affine_transform` uses this value as the upper bound for the
# OPE range.
MAX_NUM_OPE_ENC = DEFAULT_IN_RANGE_END


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

    def __init__(self, model, test_data_min, test_data_max):
        """
        :param model: plaintext PPXGBoost model
        :param test_data_min: minimum value across all features of all (future) queries
        :param test_data_max: maximum value across all features of all (future) queries
        """
        model_min, model_max = model.get_extreme_values()
        self.min_val = min(test_data_min, model_min)
        self.max_val = max(test_data_max, model_max)

    def affine_transform(self, x):
        """
        This affine transformation will linearly rescale [min_val, max_val] to [0, MAX_NUM_OPE_ENC].
        Linear rescaling:  (x - min_val) * MAX_NUM_OPE_ENC / (max_val - min_val)

        :param x: float in range [min_val, max_val]
        :return: int in range [0, MAX_NUM_OPE_ENC]
        """
        return int((x - self.min_val) * MAX_NUM_OPE_ENC / (self.max_val - self.min_val))


# pytest will try to do weird things if you have a function name that starts with `test_`!!
# "fixture 'test_data' not found"
def get_test_data_extreme_values(test_data):
    """
    Extract the minimum and maximum values from a list of queries.

    The `OPEMetadata` class requires an estimate for the minimum and maximum
    values across all features and all queries. For tests and examples,
    the entire test dataset is known in advance, so we can just extract
    the minimum and maximum values from the dataset directly.

    :param test_data: A non-empty list of queries (dictionaries)
    :return: minimum value across all queries, maximum value across all queries
    """

    def query_extremes(q):
        values = q.values()
        return min(values), max(values)

    extremes = map(query_extremes, test_data)
    # unpack the list of tuples into mins and maxes
    q_mins, q_maxs = list(zip(*extremes))

    return min(q_mins), max(q_maxs)
