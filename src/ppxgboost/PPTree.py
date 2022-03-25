# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import regualr expression
import re
import numpy as np
import pandas as pd
from ppxgboost.PPKey import *
from ppxgboost.OPEMetadata import *
from ppxgboost.PPBooster import *

# The precision bound we cannot tolerate (beyond this we cannot handle it)
PRECISION_BOUND_COMP_ZERO = 1.0e-8
# We set the precision to the following bound
SETUP_BOUND_COMP_ZERO = 1.0e-7

# This module implements a basic tree structure for XGBoost trees,
# plus serialization to/deserialization from a string. The serialization
# functionality allows us to import models from the xgboost library.

class TreeNode:

    # identifier: a unique value identifying this node
    def __init__(self, identifier):
        self.id = identifier

# A leaf of a tree
class Leaf(TreeNode):

    # identifier: a unique value identifying this node
    # value is the score.
    def __init__(self, identifier, value):
        super().__init__(identifier)
        self.value = value

    # ignore input, this is a leaf
    def eval(self, x):
        return self.value

    # Serialize this leaf as a string at level `lvl`. Note that this
    # serialization is carefully designed to match the xgboost.get_dump()
    # so that the deserialization function works for both model types.
    def node_to_string(self, lvl):
        ans = ""
        for i in range(0, lvl):
            ans = ans + "\t"
        return ans + str(self.id) + ":leaf=" + str(self.value) + "\n"

    # Get a set of all features used in this XGBoost tree
    def get_features(self):
        return set()

    # Get the minimum and maximum values used for comparison in the tree
    # This metadata is needed when selecting encryption parameters.
    def get_extreme_values(self):
        max_val = float('-inf')
        min_val = float('inf')
        return min_val, max_val

    # As Client's query is unpredictible, it's impossible
    # this is only to avoid the comparision between two extreme tiny
    # values when encountered in the model. The Affine transform from
    # PPBoost.py already takes care of the precision up to 1.0e-7.
    #
    # In case we encountered a comparision
    # between ~1.0e-14 and 0. OPE cannot support this,
    # so manually set the tiny number (1.0e-14) to a bigger
    # number (1.0e-7) in order to make the comparison go thru.
    # As a result, the current methodology cannot support more than 7 digits of
    # floating number precision.
    def discretize(self):
        if abs(self.value) <= PRECISION_BOUND_COMP_ZERO:
            self.value = SETUP_BOUND_COMP_ZERO * int(np.sign(self.value))

    def encrypt(self, pp_boost_key: PPBoostKey, metadata: Metadata):
        encrypted_value = paillier.encrypt(pp_boost_key.get_public_key(), self.value)
        return Leaf(self.id, encrypted_value)


# An interior node in the tree data structure.
class Interior(TreeNode):

    # identifier: a unique value identifying this node
    # feature_name: the name of the feature to compare
    # cmp_val: the value to compare for the feature. All comparisons are "x < cmp_val".
    # if_true_child: the node to go to if the comparision statement is true.
    # if_false_child: the node to go to if the comparision statement is false.
    # default_child: the node to go to if the feature is missing from the data.
    def __init__(self, identifier, feature_name, cmp_val, if_true_child, if_false_child, default_child):
        super().__init__(identifier)
        self.feature_name = feature_name
        self.cmp_val = cmp_val

        if (if_true_child != default_child and if_false_child != default_child):
            raise Exception("Default child must be either the 'true' child or the 'false' child")

        self.if_true_child = if_true_child
        self.if_false_child = if_false_child
        self.default_child = default_child

    # the evaluation of the tree.
    def eval(self, x):
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

    # Serialize this tree as a string starting at level `lvl`. Note that this
    # serialization is carefully designed to match the xgboost.get_dump()
    # so that the deserialization function works for both model types.
    def node_to_string(self, lvl):
        ans = ""
        for i in range(0, lvl):
            ans = ans + "\t"

        ans = "{}{}:[{}<{}] yes={},no={},missing={}\n".format(ans, str(self.id), str(self.feature_name),
                                                              str(self.cmp_val),
                                                              str(self.if_true_child.id), str(self.if_false_child.id),
                                                              str(self.default_child.id))

        return ans + self.if_true_child.node_to_string(lvl + 1) + self.if_false_child.node_to_string(lvl + 1)

    # Get a set of all features used in this XGBoost tree
    def get_features(self):
        feature_set = set()
        feature_set.add(self.feature_name)
        feature_set = feature_set.union(self.if_true_child.get_features())
        feature_set = feature_set.union(self.if_false_child.get_features())
        return feature_set

    # Get the minimum and maximum values used for comparison in the tree
    # This metadata is needed when selecting encryption parameters.
    def get_extreme_values(self):
        min1, max1 = self.if_true_child.get_extreme_values()
        min2, max2 = self.if_false_child.get_extreme_values()
        return min(min1, min2, self.cmp_val), max(max1, max2, self.cmp_val)

    def discretize(self):
        if abs(self.cmp_val) <= PRECISION_BOUND_COMP_ZERO:
            self.cmp_val = SETUP_BOUND_COMP_ZERO * int(np.sign(self.cmp_val))
        self.if_true_child.discretize()
        self.if_false_child.discretize()

    def encrypt(self, pp_boost_key: PPBoostKey, metadata: Metadata):
        num = metadata.affine_transform(self.cmp_val)
        if num > MAX_NUM_OPE_ENC or num < 0:
            raise Exception("Invalid input: input is out of range (0, " + MAX_NUM_OPE_ENC +
                            "), system cannot encrypt", num)
        encrypted_val = pp_boost_key.get_ope_encryptor().encrypt(num)

        # TODO: we end up recomputing HMACs many times, which may be slowing encryption down. Cache?
        # print("TreeEnc: HMAC of " + self.feature_name + " is " + hmac_code)
        encrypted_feature = hmac_msg(pp_boost_key.get_prf_key(), self.feature_name)

        encrypted_true_subtree = self.if_true_child.encrypt(pp_boost_key, metadata)
        encrypted_false_subtree = self.if_false_child.encrypt(pp_boost_key, metadata)

        if self.if_true_child == self.default_child:
            encrypted_default_subtree = encrypted_true_subtree
        else:
            encrypted_default_subtree = encrypted_false_subtree

        return Interior(self.id, encrypted_feature, encrypted_val, encrypted_true_subtree, encrypted_false_subtree, encrypted_default_subtree)


# Create a string representation of the tree. This should be the same as the xgboost model dump for the tree.
def tree_to_string(t: TreeNode):
    return t.node_to_string(0)




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
    # The index of the regular expression is defined based on the xgboost
    # output formatting.

    # split the string into chunks where each represents a single tree_node
    # the first item in the list is the root for this tree
    current_node = re.split(r"\n", s)[0]

    # try to parse this tree_node as a leaf
    # a leaf is identified by the pattern <int>:leaf=<float>
    # where the float is possibly negative and possibly an integer
    # # similar to '\w.-'. The '-' is to capture negative value, and '.' for floating point number.
    # e.g. 10:leaf=0.009 will be parsed to ['10', '0.009']
    leaf_strs = re.findall(r"[\d.-]+", current_node)

    if len(leaf_strs) == 2:  # if this is a leaf
        return Leaf(int(leaf_strs[0]), float(leaf_strs[1]))
    else:
        # the parsing for scientific notations.
        # if the leaf value is a scientific notation, then the parsing above does not work
        # the value contains 'e-', e.g. '2.3e-05', therefore, we need to parsing the scientific value
        if len(leaf_strs) == 3 and 'e-' in current_node:
            pos = current_node.find('=')
            value = float(current_node[pos + 1:])

            # ericcro TODO: I don't understand what this is supposed to accomplish.
            # Certainly it's true that the OPE has limited precision, but I'm disinclined to
            # handle that here; it should maybe be done at encryption time if it needs to be
            # done at all. Handling precision here breaks the property that
            #     t == parse_subtree(tree_to_string(t))
            # I'm also unsure why we're applying the precision change to both
            # OPE- and Paillier-encrypted values; there doesn't seem to be any reason to use
            # the same precision for both schemes.
            # Finally, I don't understand why we discretize when we find \eps, but not,
            # e.g., 1+\eps.
            #
            #
            # As Client's query is unpredictible, it's impossible
            # this is only to avoid the comparision between two extreme tiny
            # values when encountered in the model. The Affine transform from
            # PPBoost.py already takes care of the precision up to 1.0e-7.
            #
            # In case we encountered a comparision
            # between ~1.0e-14 and 0. OPE cannot support this,
            # so manually set the tiny number (1.0e-14) to a bigger
            # number (1.0e-7) in order to make the comparison go thru.
            # As a result, the current methodology cannot support more than 7 digits of
            # floating number precision.
            if abs(value) <= PRECISION_BOUND_COMP_ZERO:
                print("LEAF DISCRETIZATION")
                value = SETUP_BOUND_COMP_ZERO * int(np.sign(value))
            return Leaf(int(leaf_strs[0]), value)

    # An interior tree_node is identified by the pattern
    # '\w' means find all word characters - e.g. matches a "word" character: a letter or digit
    #   or underbar [a-zA-Z0-9_] (Note that \w contains underscore)
    # '.' and '-' are literal '.' and '-' --> add to to capture if some feature name contains '-' or '.'
    # The square bracket is used to indicate a set of characters. '+' indicates the repetitions,
    # and () indicates a match group.
    # The regex below splits the statement (i.e. tree tree_node) into
    # a list of strings. We match the column name with `[\w\s.-]+`, which allow for alpha-numeric
    # characters, whitespace, `.`, and `-`.
    #
    # e.g. current_node = '0:[XYZ ABC<3] yes=1,no=2,missing=1'
    # leaf_strs = ['0', 'XYZ ABC', '3', 'yes', '1', 'no', '2', 'missing', '1']
    pattern = re.compile(r"(\w+):\[([\w\s.-]+)[<>=]+(.+)\] (\w+)=(.+),(\w+)=(.+),(\w+)=(.+)")
    match = pattern.match(current_node)
    if match is None:
        raise Exception("Invalid tree:\n" + current_node + "\nNote that column names can only contain alpha-numeric characters, whitespace, '.', or '-'.")
    leaf_strs = match.groups()

    # we've parsed the root, now find and parse the subtrees
    split_str = r"\n"
    for i in range(0, lvl + 1):
        split_str = split_str + r"\t"

    # split the input on \n\t...\t[0-9]
    # This splits the string into 5 pieces:
    # index 0 is current_node,
    # index 1 is the id of the left subtree
    # index 2 is the rest of the string for the left subtree
    # index 3 is the id of the right subtree
    # index 4 is the rest of the string for the right subtree
    subtrees = re.split(split_str + r"(\d+)", s)

    # recurse to the next level.
    left = parse_subtree(subtrees[1] + subtrees[2], lvl + 1)
    right = parse_subtree(subtrees[3] + subtrees[4], lvl + 1)

    # create a dictionary that maps the subtree Id to the subtree object
    child_dict = {left.id: left, right.id: right}

    # Check if the comparison is a floating point number.
    #   if it is then convert it to float
    #   else we convert it to an int.
    if '.' in leaf_strs[2]:
        node_value = float(leaf_strs[2])

        # Similar to above (precision issue)
        if abs(node_value) <= PRECISION_BOUND_COMP_ZERO:
            print("NODE DISCRETIZATION")
            node_value = SETUP_BOUND_COMP_ZERO * int(np.sign(node_value))
    else:
        node_value = int(leaf_strs[2])

    return Interior(int(leaf_strs[0]), leaf_strs[1], node_value, child_dict[int(leaf_strs[4])],
                    child_dict[int(leaf_strs[6])], child_dict[int(leaf_strs[8])])


# Recursively parse the tree.
def parse_tree(t):
    return parse_subtree(t, 0)
