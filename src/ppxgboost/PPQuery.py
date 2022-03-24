# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import hmac
import hashlib
import random
import string
import math

import numpy as np

from ppxgboost.OPEMetadata import *
import ppxgboost.PaillierAPI as paillier
from ppxgboost.Model import *
from ppxgboost.Tree import *

# multiplicative factor for encoding the message when using OPE
from ope.pyope.ope import DEFAULT_IN_RANGE_END
from ppxgboost.PPKey import PPBoostKey

import encodings

# hmac the msg using utf-8 encoding (python3)
# we use hmac as a PRF to create 'pseudonyms' for the features.
# A reference that shows hmac is a PRF (see Theorem 1 in https://eprint.iacr.org/2014/578.pdf)
# use hmac to instaniate prf.
def hmac_msg(prf_key_hash, feature):
    """
    Using hmac to produce the pseudonyms for the feature vector.
    :param prf_key_hash: hash key as bytes
    :param feature: feature name as a string (encoded using 'UTF-8')
    :return: hmac value
    """
    message = bytes(feature, encodings.utf_8.getregentry().name)
    sig = base64.b64encode(hmac.new(prf_key_hash, message, hashlib.sha256).digest())
    return sig.decode()

class QueryEncryptor:

    def __init__(self, client_key: ClientKey, feature_set, metadata: Metadata):
        self.key = client_key
        self.feature_set = feature_set
        self.metadata = metadata
        self.feature_name_map = {}
        for f in self.feature_set:
            self.feature_name_map[f] = hmac_msg(self.key.get_prf_key(), f)

    def encrypt_query(self, query_dict):
        encrypted_query = {}
        for k, v in query_dict.items():
            if k not in self.feature_set:
                continue

            # TODO: datasets in the tests *do* contain NaN
            # but this leaks where NaNs are...
            if math.isnan(v):
                encrypted_val = v
            else:
                normalized_val = self.metadata.affine_transform(v)

                if normalized_val > MAX_NUM_OPE_ENC or normalized_val < 0:
                    raise Exception("Invalid input: input is out of range (0, " + str(MAX_NUM_OPE_ENC) +
                                    "). The system cannot encrypt", normalized_val)
                encrypted_val = self.key.get_ope_encryptor().encrypt(int(normalized_val))
            encrypted_query[self.feature_name_map[k]] = encrypted_val
        return encrypted_query

def pandas_to_queries(data_set):
    """
    Process the feature's name using hmac, then encrypts the neccessary values using OPE
    based on the feature set.
    :param metadata: encryption metadata containing min, max and affine transform
    :param hash_key: hmac hash key
    :param ope: ope object
    :param feature_set: feature set
    :param input_vector: input vector
    """
    # starts to encrypt using ope based on the feature set.

    queries = []
    for i, row in data_set.iterrows():
        query = {}
        for feature in list(data_set.columns.values):
            query[feature] = row[feature]
        queries.append(query)
    return queries






# # encrypts the input vector
# def enc_input_vector(client_key: ClientKey, feature_set, input_vector, metadata):
#     """
#     Process the feature's name using hmac, then encrypts the neccessary values using OPE
#     based on the feature set.
#     :param metadata: encryption metadata containing min, max and affine transform
#     :param hash_key: hmac hash key
#     :param ope: ope object
#     :param feature_set: feature set
#     :param input_vector: input vector
#     """
#     # starts to encrypt using ope based on the feature set.

#     hash_key = client_key.get_prf_key()
#     ope = client_key.get_ope_encryptor()

#     feature_list = list(feature_set)

#     # Drop columns not in the feature list (i.e., columns not used by the model)
#     for col in input_vector.columns:
#         if col not in feature_list:
#             input_vector.drop(col, axis=1, inplace=True)

#     # calls the hmac_feature method to hmac the feature name of the input vector.
#     hmac_feature(hash_key, input_vector)

#     for i, row in input_vector.iterrows():
#         # Encrypts all the features in the input_vector
#         for feature in list(input_vector.columns.values):

#             # TODO: datasets in the tests *do* contain NaN
#             # but this leaks where NaNs are...
#             if math.isnan(row[feature]):
#             #     raise Exception("Found NaN in input vector")
#                 continue

#             noramlized_feature = metadata.affine_transform(row[feature])

#             if noramlized_feature > MAX_NUM_OPE_ENC or noramlized_feature < 0:
#                 raise Exception("Invalid input: input is out of range (0, " + MAX_NUM_OPE_ENC +
#                                 "). The system cannot encrypt", noramlized_feature)

#             ope_value = ope.encrypt(int(noramlized_feature))

#             input_vector.at[i, feature] = ope_value

# hmac the features for the testing vector
# def hmac_feature(prf_hash_key, input_vector):
#     """
#     hmac vector's column name
#     :param prf_hash_key: hash key
#     :param input_vector: the vector (as dataframe)
#     :return:
#     """
#     new_header = list()
#     for col in input_vector.columns:
#         new_header.append(hmac_msg(prf_hash_key, col))
#     # Reassign the column names to the input vector
#     input_vector.columns = new_header
#     return input_vector


# # This method recursively encrypts the tree_node comparison value using OPE scheme
# # It also uses the PRF to 'pseudo-randomize' the feature name as well
# # It then encrypts the leaf value using he_pub_key (he public key).
# def enc_tree_node(he_pub_key, prf_hash_key, ope, tree_node, metadata):
#     """
#     Process the node
#     :param metadata:
#     :param he_pub_key: the homomorphic key
#     :param prf_hash_key: hash key for hmac
#     :param ope: ope object for encrypting the comparison value
#     :param tree_node: the Interier object (node) in the decision tree.
#     :return: ope encrypted tree
#     """
#     # If it is not leaf, then encode the comp_val using OPE.
#     if not isinstance(tree_node, Leaf):

#         num = metadata.affine_transform(tree_node.cmp_val)

#         if num > MAX_NUM_OPE_ENC or num < 0:
#             raise Exception("Invalid input: input is out of range (0, " + MAX_NUM_OPE_ENC +
#                             "), system cannot encrypt", num)

#         tree_node.cmp_val = ope.encrypt(num)

#         hmac_code = hmac_msg(prf_hash_key, tree_node.feature_name)
#         # TODO: we end up recomputing HMACs many times, which may be slowing encryption down. Cache?
#         # print("TreeEnc: HMAC of " + tree_node.feature_name + " is " + hmac_code)
#         tree_node.feature_name = hmac_code

#         # Recurse to the if true tree_node
#         enc_tree_node(he_pub_key, prf_hash_key, ope, tree_node.if_true_child, metadata)
#         # Recurse to the if false tree_node
#         enc_tree_node(he_pub_key, prf_hash_key, ope, tree_node.if_false_child, metadata)
#     # else it is the Leaf
#     else:
#         # Value....
#         tree_node.value = paillier.encrypt(he_pub_key, tree_node.value)


# def enc_xgboost_model(ppBoostKey: PPBoostKey, model: XGBoostModel, metadata: Metadata):
#     """
#     Encrypts the model to an encrypted format.
#     :param ppBoostKey: the pp boost key wrapper.
#     :param metadata: metadata containing min, max information
#     :param trees: the model as an input (a list of trees)
#     """
#     trees = model.trees
#     he_pub_key = ppBoostKey.get_public_key()
#     prf_hash_key = ppBoostKey.get_prf_key()
#     ope = ppBoostKey.get_ope_encryptor()

#     for t in trees:
#         enc_tree_node(he_pub_key, prf_hash_key, ope, t, metadata)
#     return trees

