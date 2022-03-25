# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ppxgboost.PPTree import PPTree
from ppxgboost.PPKey import *
from ppxgboost.OPEMetadata import *

class PPModel:
    """
    A representation of an XGBoost model composed of multiple trees.
    """

    def __init__(self, trees: list[PPTree]):
        """
        :param trees: a list of PPTrees
        """

        self.trees = list(trees)

    def eval(self, x):
        """
        Evaluate the model on the given query.

        :param x: dictionary corresponding to the input
        :return: result of evaluating the model on x
        """

        return map(lambda t: t.eval(x), self.trees)

    def get_features(self):
        """
        :return: The set of features used by this model
        """

        features = set()
        for t in self.trees:
            features = features.union(t.get_features())
        return features

    def get_extreme_values(self):
        """
        :return: The minimum and maximum comparison values in the model
        """

        min_val = float('inf')
        max_val = float('-inf')
        for t in self.trees:
            t_min, t_max = t.get_extreme_values()
            min_val = min(min_val, t_min)
            max_val = max(max_val, t_max)
        return min_val, max_val

    def discretize(self):
        """
        Discretize the model in preparation for encrypted evaluation.
        Do this step before encryption.

        :return: None
        """

        for t in self.trees:
            t.discretize()

    def encrypt(self, pp_boost_key: PPModelKey, metadata: OPEMetadata):
        """
        Encrypt a plaintext XGBoost model

        :param pp_boost_key: The model encryption key
        :param metadata: OPE metadata
        :return: a new PPModel corresponding to the encryption of `self`
        """
        return XGBoostModel(map(lambda t: t.encrypt(pp_boost_key, metadata), self.trees))


def from_xgboost_model(model):
    """
    Create a PPModel from an xgboost library model.

    :param model: an XGBoost model from the xgboost library
    :return: a PPModel corresponding to the input model
    """

    # xgboost doesn't provide a way to access the model parameters directly,
    # so we convert to the PPModel representation via serialization.
    # Specifically, xgboost will return a string representation of the model,
    # which we then convert to a PPModel using PPTree deserialization. This
    # works because the PPTree serialization was designed to coincide with
    # xgboost serialization of a model.

    # get_dump() returns a string representation of the xgboost model.
    # We parse this string representation one tree at a time to create
    # an internal representation of the model.
    trees_dump = model.get_dump()

    # For each string representation of an xgboost tree, parse the representation
    # into an internal representation of the tree
    output_trees = map(Tree.parse_tree, trees_dump)

    # output an (internal) XGBoost model
    return XGBoostModel(output_trees)
