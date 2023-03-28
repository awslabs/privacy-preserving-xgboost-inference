# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import re

from ppxgboost.PPKey import *
# from ppxgboost.OPEMetadata import *
# from ppxgboost.PPQuery import hmac_msg

# The precision bound we cannot tolerate (beyond this we cannot handle it)
PRECISION_BOUND_COMP_ZERO = 1.0e-8
# We set the precision to the following bound
SETUP_BOUND_COMP_ZERO = 1.0e-7

# This module implements a basic tree structure for XGBoost trees,
# plus serialization to/deserialization from a string. The serialization
# functionality allows us to import models from the xgboost library.

class TreeNode:
    """
    A parent class for interior and leaf nodes of a tree
    """

    def __init__(self, identifier):
        """
        :param identifier: a unique value identifying this node within the tree
        """

        self.id = identifier


class Leaf(TreeNode):
    """
    A leaf node of a tree
    """

    def __init__(self, identifier, value):
        """
        :param identifier: a unique value identifying this node within the tree
        :param value: score for this leaf node
        """

        super().__init__(identifier)
        self.value = value

    def eval(self, _):
        """
        Evaluate the model on an input. Since this is a leaf node,
        we ignore the input and return the node's value.

        :return: result of evaluating the model on the provided query
        """

        return self.value

    def node_to_string(self, lvl):
        """
        Serialize this leaf as a string at level `lvl`. Note that this
        serialization is carefully designed to match the xgboost.get_dump()
        so that the deserialization function works for both model types.

        :param lvl: The level in the tree of this leaf
        :return: A string representing this leaf node
        """

        return '\t'*lvl + str(self.id) + ":leaf=" + str(self.value) + "\n"

    def get_features(self):
        """
        :return: The set of all features used in this XGBoost tree
        """

        return set()

    def get_extreme_values(self):
        """
        :return: the minimum and maximum values used for comparison in the tree
        """

        return float('inf'), float('-inf')

class Interior(TreeNode):
    """
     An interior node in a (binary) tree
    """

    def __init__(self, identifier, feature_name, cmp_val, if_true_child, if_false_child, default_child):
        """
        :param identifier: a unique value identifying this node within the tree
        :param feature_name: the name of the feature to compare
        :param cmp_val: the value to compare for the feature. All comparisons are "x < cmp_val".
        :param if_true_child: the node to go to if the comparision statement is true.
        :param if_false_child: the node to go to if the comparision statement is false.
        :param default_child: the node to go to if the feature is missing from the data. This MUST be
                              either if_true_child or if_false_child
        """

        super().__init__(identifier)
        self.feature_name = feature_name
        self.cmp_val = cmp_val

        if (if_true_child != default_child and if_false_child != default_child):
            raise Exception("Default child must be either the 'true' child or the 'false' child")

        self.if_true_child = if_true_child
        self.if_false_child = if_false_child
        self.default_child = default_child

    def eval(self, x):
        """
        Evaluate the model on the given query.

        :param x: dictionary corresponding to the input
        :return: result of evaluating the model on x
        """

        if x is None:
            raise RuntimeError("None in eval")
        if self.feature_name not in x:
            print('Feature name ' + self.feature_name + ' is not available in query')
            print(x)
            raise RuntimeError("Feature name not available in eval")

        if np.isnan(x[self.feature_name]):
            return self.default_child.eval(x)

        if x[self.feature_name] < self.cmp_val:
            return self.if_true_child.eval(x)
        else:
            return self.if_false_child.eval(x)

    def node_to_string(self, lvl):
        """
        Serialize this tree as a string at level `lvl`. Note that this
        serialization is carefully designed to match the xgboost.get_dump()
        so that the deserialization function works for both model types.

        :param lvl: The level in the tree of this node
        :return: A string representing this tree node
        """

        ans = ""
        for i in range(0, lvl):
            ans = ans + "\t"

        ans = "{}{}:[{}<{}] yes={},no={},missing={}\n".format(ans, str(self.id), str(self.feature_name),
                                                              str(self.cmp_val),
                                                              str(self.if_true_child.id), str(self.if_false_child.id),
                                                              str(self.default_child.id))

        return ans + self.if_true_child.node_to_string(lvl + 1) + self.if_false_child.node_to_string(lvl + 1)

    def get_features(self):
        """
        :return: The set of features used by this model
        """

        feature_set = set()
        feature_set.add(self.feature_name)
        feature_set = feature_set.union(self.if_true_child.get_features())
        feature_set = feature_set.union(self.if_false_child.get_features())
        return feature_set

    def get_extreme_values(self):
        """
        :return: The minimum and maximum comparison values in the model
        """

        min1, max1 = self.if_true_child.get_extreme_values()
        min2, max2 = self.if_false_child.get_extreme_values()
        return min(min1, min2, self.cmp_val), max(max1, max2, self.cmp_val)

def tree_to_string(t: TreeNode):
    """
    Serialize this tree as a string. Note that this
    serialization is carefully designed to match the xgboost.get_dump()
    so that the deserialization function works for both model types.

    :param t: the root of the tree
    :return: a string representing this tree node
    """

    return t.node_to_string(0)


# Create a PPXGBoost model from a string serialization. PPXGBoost requires
# granular access to model parameters, but the xgboost library doesn't
# provide this level of visibility directly. Instead, we designed
# serialization of PPXGBoost models so that it matches the way that
# xgboost serializes models via `get_dump()`. See
# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_dump
# for details. Thus this function can parse
# xgboost trees serialized via `get_dump()` _or_ PPXGBoost trees serialized
# via `tree_to_string()`.
def parse_subtree(s, lvl):
    """
    Parse a subtree starting at a specific level.

    :param s: string representing the subtree
    :param lvl: level of this subtree in the full tree
    :return: a TreeNode representing this subtree
    """

    # An example tree serialization string is:
    #
    #  '0:[XXX<3] yes=1,no=2,missing=1\n\t1:[Fare<13.6458502] yes=3,no=4,missing=3\n\t\t
    #  3:leaf=-0.00585523667\n\t\t4:leaf=0.0201724116\n\t2:leaf=-0.0114313215\n
    #
    # This represents the following tree structure:
    #  0:[XXX<3] yes=1,no=2,missing=1
    #    1:[xyz<13.6458502] yes=3,no=4,missing=3
    #      3:leaf=-0.00585523667
    #      4:leaf=0.0201724116
    #    2:leaf=-0.0114313215

    # Recall that serialization (and therefore deserialization) are based on the
    # xgboost library's serialization approach.

    # split the string into chunks where each represents a single tree_node
    # the first item in the list is the root for this tree
    current_node = re.split(r'\n', s)[0]

    # a regular expression for parsing a variety of numeric formats
    # -?: an optional leading '-' sign
    # (\d*\.\d+|\d+): zero or more digits followed by a decimal and one or more digits
    #                 _or_ one or more digits (an integer). Order appears to matter here!
    #                 If the integer portion is first, a float that contains a whole number
    #                 component will match to it instead of the float clause.
    # (e-?\d+)?: optional scientific notation suffix. The exponent has an optional
    #            '-' sign followed by one or more digits
    number_regex = r'-?(\d*\.\d+|\d+)(e-?\d+)?'

    # a regex for parsing feature names
    # '\w' means find all word characters - e.g. matches a "word" character: a letter or digit
    # or underbar [a-zA-Z0-9_] (Note that \w contains underscore)
    # '\s' means whitespace
    # '.' and '-' are literal characters
    # Thus, a feature name is a sequence of one or more of these characters.
    feature_name_regex = r'[\w\s.-]+'

    # a regular expression for parsing a leaf node, which has the pattern '<int>:leaf=<float>'
    leaf_regex_str = r'(?P<id>\d+):leaf=(?P<val>' + number_regex + ')'

    leaf_node_regex = re.compile(leaf_regex_str)

    # a regex to parse an interior node, which has the pattern like '0:[XYZ ABC<3] yes=1,no=2,missing=1'
    interior_node_regex = re.compile(r'(?P<id>\d+):\[(?P<feature>' + feature_name_regex + r')<(?P<cmp_val>' + number_regex + r')] yes=(?P<true>\d+),no=(?P<false>\d+),missing=(?P<default>\d+)')

    # convert a string to a numeric type
    # Unfortunately, we can't just use `float()` because
    # if the value is an integer in the string, it would
    # be serialized as a float (i.e., with a trailing '.0')
    # making the identity test fail
    def val_to_num(val):
        if val.isdigit():
            return int(val)
        else:
            return float(val)

    # match current node against the two patterns
    leaf_regex_match = leaf_node_regex.match(current_node)
    if leaf_regex_match is not None:
        return Leaf(int(leaf_regex_match.group('id')), val_to_num(leaf_regex_match.group('val')))

    interior_regex_match = interior_node_regex.match(current_node)
    if interior_regex_match is None:
        raise Exception("Invalid tree:\n" + current_node + "\nNote that column names can only contain alpha-numeric characters, whitespace, '.', or '-'.")

    # otherwise, we have successfully parsed an interior node


    # we've parsed the root, now find and parse the subtrees
    # subtrees are at level (lvl+1), which has a prefix with (lvl+1) tabs
    split_str = r'\n' + r'\t'*(lvl + 1)

    # Subtree matching extracts the left and right subtrees of current_node.
    # The '.+' matches current_node.
    # 'split_str' matches the subtree delimiter
    # '\d(.|\n)+' _ends_ the subtree delimiter (to avoid matching too many tabs for this level),
    #             then matches one or more '.' (any char except \n) or \n
    # 'split_str' matches the second subtree delimiter
    # '\d(.|\n)+' _ends_ the subtree delimiter (to avoid matching too many tabs for this level),
    #             then matches one or more '.' (any char except \n) or \n
    subtree_regex = re.compile(r'.+' + split_str + r'(?P<left_subtree>\d(.|\n)+)' + split_str + r'(?P<right_subtree>\d(.|\n)+)')
    subtree_match = subtree_regex.match(s)

    if subtree_match is None:
        raise Exception('invalid subtree structure\n' + repr(split_str) + '\n' + repr(s))

    # recurse to the next level.
    left_subtree = parse_subtree(subtree_match.group('left_subtree'), lvl + 1)
    right_subtree = parse_subtree(subtree_match.group('right_subtree'), lvl + 1)

    # create a dictionary that maps the subtree Id to the subtree object
    child_dict = {left_subtree.id: left_subtree, right_subtree.id: right_subtree}

    return Interior(int(interior_regex_match.group('id')),
                    interior_regex_match.group('feature'),
                    val_to_num(interior_regex_match.group('cmp_val')),
                    child_dict[int(interior_regex_match.group('true'))],
                    child_dict[int(interior_regex_match.group('false'))],
                    child_dict[int(interior_regex_match.group('default'))])


def parse_tree(tree_string):
    """
    Parse a tree from its string serialization

    :param tree_string: string representation of the tree
    :return: a TreeNode for the root of the tree
    """

    return parse_subtree(tree_string, 0)
