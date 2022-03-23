# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import regualr expression
import re
import numpy as np
import pandas as pd
from ppxgboost.Tree import *
from ppxgboost.Model import *

# Although it would be nice if the model encryption was context-free, the OPE
# scheme needs to be aware of the min and max values it will need to encrypt.
# This function returns the min and max values over the test data set to use
def training_dataset_parser(train_data: pd.DataFrame):
    """
    :param train_data: dataframe training data
    :return: minimum of the training dataset, and maximum of the training dataset.
    """
    return {'min': np.min(pd.DataFrame.min(train_data)), 'max': np.max(pd.DataFrame.max(train_data))}
