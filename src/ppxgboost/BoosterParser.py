# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import regualr expression
import ppxgboost.PPModel as PPModel
import xgboost
import numpy as np
import pandas as pd

# The function parses the pickle file to a model (xgboost)
def model_to_trees(model: xgboost.core.Booster, min_max):
    """
    Parse the model to trees
    :param min_max: dictionary key {'min','max'}
            min_max['min'] min_max['max']
    :param model: the xgboost model
    :return: the parse tree, the features in the xgboost model
    """
    # getting the dump of the tree.
    # list of string (representing trees)
    # the get_dump() returns a list strings, each tree is represented in a particular format
    # (seperated by \n or \t's.
    #  For example: one of the tree' string representation is below:
    #  '0:[XXX<3] yes=1,no=2,missing=1\n\t1:[Fare<13.6458502] yes=3,no=4,missing=3\n\t\t
    #  3:leaf=-0.00585523667\n\t\t4:leaf=0.0201724116\n\t2:leaf=-0.0114313215\n
    # -->
    # represents the following tree structure.
    # 0:[XXX<3] yes=1,no=2,missing=1
    #   1:[xyz<13.6458502] yes=3,no=4,missing=3
    # 	    3:leaf=-0.00585523667
    # 	    4:leaf=0.0201724116
    #   2:leaf=-0.0114313215

    ppmodel = PPModel.from_xgboost_model(model)
    model_min, model_max = ppmodel.get_extreme_values()
    min_max['min'] = min(min_max['min'], model_min)
    min_max['max'] = max(min_max['max'], model_max)

    # output a list of the tree objects.
    return ppmodel, ppmodel.get_features(), min_max


def training_dataset_parser(train_data: pd.DataFrame):
    """
    :param train_data: dataframe training data
    :return: minimum of the training dataset, and maximum of the training dataset.
    """
    return {'min': np.min(pd.DataFrame.min(train_data)), 'max': np.max(pd.DataFrame.max(train_data))}
