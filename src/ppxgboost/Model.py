
# An XGBoost model is a collection of TreeNodes

import ppxgboost.Tree as Tree
from ppxgboost.PPKey import *
from ppxgboost.OPEMetadata import *

class XGBoostModel:

    def __init__(self, trees):
        self.trees = list(trees)

    def eval(self, x):
        return map(lambda t: t.eval(x), self.trees)

    def get_features(self):
        features = set()
        for t in self.trees:
            features = features.union(t.get_features())
        return features

    def get_extreme_values(self):
        min_val = float('inf')
        max_val = float('-inf')
        for t in self.trees:
            t_min, t_max = t.get_extreme_values()
            min_val = min(min_val, t_min)
            max_val = max(max_val, t_max)
        return min_val, max_val

    def discretize(self):
        for t in self.trees:
            t.discretize()

    def encrypt(self, pp_boost_key: PPBoostKey, metadata: Metadata):
        """ Return a new XGBoostModel corresponding to the encryption of
            self under pp_boost_key
        """
        return XGBoostModel(map(lambda t: t.encrypt(pp_boost_key, metadata), self.trees))


# Converts an xgboost model to an internal representation of the model.
# Returns the internal representation of the model
def from_xgboost_model(model):
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
    #       3:leaf=-0.00585523667
    #       4:leaf=0.0201724116
    #   2:leaf=-0.0114313215

    # get_dump() returns a string representation of the xgboost model.
    # We parse this string representation one tree at a time to create
    # an internal representation of the model.
    trees_dump = model.get_dump()

    # For each string representation of an xgboost tree, parse the representation
    # into an internal representation of the tree
    output_trees = map(Tree.parse_tree, trees_dump)

    # output an (internal) XGBoost model
    return XGBoostModel(output_trees)
