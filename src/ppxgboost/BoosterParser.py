# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import regualr expression
import ppxgboost.PPModel as PPModel
import xgboost
import numpy as np
import pandas as pd


def training_dataset_parser(train_data: pd.DataFrame):
    """
    :param train_data: dataframe training data
    :return: minimum of the training dataset, and maximum of the training dataset.
    """
    return {'min': np.min(pd.DataFrame.min(train_data)), 'max': np.max(pd.DataFrame.max(train_data))}
