# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import regualr expression
import re
import numpy as np
import pandas as pd
from ppxgboost.Tree import *
from ppxgboost.Model import *

# Converts an xgboost model to an internal representation of the model.
# Returns the internal representation of the model as well as a list of
# features used in the model
def model_to_trees(model):
    """
    Parse the model to trees
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

    # get_dump() returns a string representation of the xgboost model.
    # We parse this string representation one tree at a time to create
    # an internal representation of the model.
    trees_dump = model.get_dump()

    # For each string representation of an xgboost tree, parse the representation
    # into an internal representation of the tree
    output_trees = map(parse_tree, trees_dump)

    # output an (internal) XGBoost model
    return XGBoostModel(output_trees)

# Although it would be nice if the model encryption was context-free, the OPE
# scheme needs to be aware of the min and max values it will need to encrypt.
# This function returns the min and max values over the test data set to use
def training_dataset_parser(train_data: pd.DataFrame):
    """
    :param train_data: dataframe training data
    :return: minimum of the training dataset, and maximum of the training dataset.
    """
    return {'min': np.min(pd.DataFrame.min(train_data)), 'max': np.max(pd.DataFrame.max(train_data))}
