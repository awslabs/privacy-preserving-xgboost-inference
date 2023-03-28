# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ppxgboost.PPTree as PPTree
from ppxgboost.PPKey import PPModelKey
import xgboost
from ppxgboost.OPEMetadata import OPEMetadata

class PPModel:
    """
    A representation of an XGBoost model composed of multiple trees.
    """

    def __init__(self, trees):
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

        return list(map(lambda t: t.eval(x), self.trees))

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

    def encrypt(self, pp_boost_key: PPModelKey, metadata: OPEMetadata):
        """
        Encrypt a plaintext XGBoost model

        :param pp_boost_key: The model encryption key
        :param metadata: OPE metadata
        :return: a new PPModel corresponding to the encryption of `self`
        """

        # use a global dictionary so that we only encrypt each feature once
        global_feature_encryption_dict = {}
        return PPModel(map(lambda t: t.encrypt(pp_boost_key, metadata, global_feature_encryption_dict), self.trees))


def from_xgboost_model(model: xgboost.core.Booster) -> PPModel:
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

    # get_dump() returns a list strings, each representing a single tree in the model
    # We parse this string representation one tree at a time to create
    # an internal representation of the model.
    trees_dump = model.get_dump()

    # For each string representation of an xgboost tree, parse the representation
    # into an internal representation of the tree
    output_trees = map(PPTree.parse_tree, trees_dump)

    # output an (internal) XGBoost model
    return PPModel(output_trees)
