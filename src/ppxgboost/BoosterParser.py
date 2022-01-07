# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# import regualr expression
import re
import numpy as np
import pandas as pd

# The precision bound we cannot tolerate (beyond this we cannot handle it)
PRECISION_BOUND_COMP_ZERO = 1.0e-8
# We set the precision to the following bound
SETUP_BOUND_COMP_ZERO = 1.0e-7


# This the a leaf data structure that holds the leaf object.
class Leaf:

    # id is the leaf state, value is the score.
    def __init__(self, id, value):
        self.id = id
        self.value = value

    # ignore input, this is a leaf
    def eval(self, x: pd.Series):
        if x is None:
            raise Exception("Invalid input string.")

        return self.value

    # print out the leaf
    def node_to_string(self, lvl):
        ans = ""
        for i in range(0, lvl):
            ans = ans + "\t"
        return ans + str(self.id) + ":leaf=" + str(self.value) + "\n"


# This the interior data structure in the tree data structure.
class Interior:

    # id: the identifier of the tree_node
    # feature_name: the feature to compare
    # cmp_val: the value to compare for the feature.
    # if_true_child: if the comparision statement is true.
    # if_false_child: if the comparision statement is false.
    # default_child: if the data[feature] is missing.
    def __init__(self, identifier, feature_name, cmp_val, if_true_child, if_false_child, default_child):
        self.id = identifier
        self.feature_name = feature_name
        self.cmp_val = cmp_val
        self.if_true_child = if_true_child
        self.if_false_child = if_false_child
        self.default_child = default_child

    # the evaluation of the tree.
    def eval(self, x: pd.Series):

        if np.isnan(x[self.feature_name]):
            return self.default_child.eval(x)

        if x[self.feature_name] < self.cmp_val:
            return self.if_true_child.eval(x)
        else:
            return self.if_false_child.eval(x)

    # print the tree_node for this level
    def node_to_string(self, lvl):
        ans = ""
        for i in range(0, lvl):
            ans = ans + "\t"

        ans = "{}{}:[{}<{}] yes={},no={},missing={}\n".format(ans, str(self.id), str(self.feature_name),
                                                              str(self.cmp_val),
                                                              str(self.if_true_child.id), str(self.if_false_child.id),
                                                              str(self.default_child.id))

        return ans + self.if_true_child.node_to_string(lvl + 1) + self.if_false_child.node_to_string(lvl + 1)


# print the tree.
def tree_to_string(t):
    return t.node_to_string(0)


# parse each tree_node of the tree based on the level.
# Each tree_node will be parse to a tree_node stored in the Tree data structure.
# The parsing recursively parse the tree_node value until we encounter the leaves.
#
# This function parses the output of https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_dump
# The XGBoost package does not provide a way to read a model based on the output of `get_dump`, and we do not
# control the format of the string, so this function provides a custom parser for that strings returned by
# `get_dump`.
def parse_node_in_tree(s, lvl, feature_set, min_max):
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
    left = parse_node_in_tree(subtrees[1] + subtrees[2], lvl + 1, feature_set, min_max)
    right = parse_node_in_tree(subtrees[3] + subtrees[4], lvl + 1, feature_set, min_max)

    # create a dictionary that maps the subtree Id to the subtree object
    child_dict = {left.id: left, right.id: right}

    # Check if the comparison is a floating point number.
    #   if it is then convert it to float
    #   else we convert it to an int.
    if '.' in leaf_strs[2]:
        node_value = float(leaf_strs[2])

        # Similar to above (precision issue)
        if abs(node_value) <= PRECISION_BOUND_COMP_ZERO:
            node_value = SETUP_BOUND_COMP_ZERO * int(np.sign(node_value))
    else:
        node_value = int(leaf_strs[2])

    if node_value < min_max['min']:
        min_max['min'] = node_value

    if node_value > min_max['max']:
        min_max['max'] = node_value

    feature_set.add(str(leaf_strs[1]))

    return Interior(int(leaf_strs[0]), leaf_strs[1], node_value, child_dict[int(leaf_strs[4])],
                    child_dict[int(leaf_strs[6])], child_dict[int(leaf_strs[8])])


# Recursively parse the tree.
def parse_tree(s, feature_set, min_max):
    return parse_node_in_tree(s, 0, feature_set, min_max)


# The function parses the pickle file to a model (xgboost)
def model_to_trees(model, min_max):
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
    trees_dump = model.get_dump()
    feature_set = set()
    # create an empty list
    output_trees = []
    # for each tree append the parsed string.

    for i in range(len(trees_dump)):
        # this parse the tree to the data structure.
        tree_object = parse_tree(trees_dump[i], feature_set, min_max)
        output_trees.append(tree_object)

    # output a list of the tree objects.
    return output_trees, feature_set, min_max


def training_dataset_parser(train_data: pd.DataFrame):
    """
    :param train_data: dataframe training data
    :return: minimum of the training dataset, and maximum of the training dataset.
    """
    return {'min': np.min(pd.DataFrame.min(train_data)), 'max': np.max(pd.DataFrame.max(train_data))}
