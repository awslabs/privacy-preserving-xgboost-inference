# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import hmac
import hashlib
import random
import string
import math

import numpy as np

import ppxgboost.BoosterParser as bs
import ppxgboost.PaillierAPI as paillier

# multiplicative factor for encoding the message when using OPE
from ope.pyope.ope import DEFAULT_IN_RANGE_END
from ppxgboost.PPKey import PPBoostKey

import encodings

# This is the maximum number that the OPE encryption can support.
#   Currently, we also set this to be the maximum number that
#   affine transform map to (see line 54).
MAX_NUM_OPE_ENC = DEFAULT_IN_RANGE_END


class MetaData:
    """
    This is a metadata structure before encryption. It contains the minimum and maximum value
    from the training dataset as well as the model file
    """

    def __init__(self, min_max: dict):
        self.mini = min_max['min']
        self.maxi = min_max['max']

    def set_min(self, new_min):
        self.mini = new_min

    def set_max(self, new_max):
        self.maxi = new_max

    def affine_transform(self, x):
        """
        This affine transformation will linearly rescale [min, max] to [0, MAX_NUM_AFFINE].
        Linear rescaling:  (x - n_min) * MAX_NUM_AFFINE / (n_max - n_min)
                           MAX_NUM_AFFINE / (n_max - n_min) x - MAX_NUM_AFFINE * n_min
        :param x: input number
        :return: mapping numerical value
        """
        return int((x - self.mini) * MAX_NUM_OPE_ENC / (self.maxi - self.mini))


def sigmoid(number):
    """
    Return the logistic function of a number
    :param number: input
    :return: 1/ (1 + e^-x)
    """
    return 1 / (1 + np.exp(-number))


def random_string(string_length=16):
    """
    generate random strings
    :param string_length: 
    :return: random strings
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(string_length))


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


# hmac the features for the testing vector
def hmac_feature(prf_hash_key, input_vector):
    """
    hmac vector's column name
    :param prf_hash_key: hash key
    :param input_vector: the vector (as dataframe)
    :return:
    """
    new_header = list()
    for col in input_vector.columns:
        new_header.append(hmac_msg(prf_hash_key, col))
    # Reassign the column names to the input vector
    input_vector.columns = new_header
    return input_vector


# This method recursively encrypts the tree_node comparison value using OPE scheme
# It also uses the PRF to 'pseudo-randomize' the feature name as well
# It then encrypts the leaf value using he_pub_key (he public key).
def enc_tree_node(he_pub_key, prf_hash_key, ope, tree_node, metaData):
    """
    Process the node
    :param metaData:
    :param he_pub_key: the homomorphic key
    :param prf_hash_key: hash key for hmac
    :param ope: ope object for encrypting the comparison value
    :param tree_node: the Interier object (node) in the decision tree.
    :return: ope encrypted tree
    """
    # If it is not leaf, then encode the comp_val using OPE.
    if not isinstance(tree_node, bs.Leaf):

        num = metaData.affine_transform(tree_node.cmp_val)

        if num > MAX_NUM_OPE_ENC or num < 0:
            raise Exception("Invalid input: input is out of range (0, " + MAX_NUM_OPE_ENC +
                            "), system cannot encrypt", num)

        tree_node.cmp_val = ope.encrypt(num)

        hmac_code = hmac_msg(prf_hash_key, tree_node.feature_name)
        tree_node.feature_name = hmac_code

        # Recurse to the if true tree_node
        enc_tree_node(he_pub_key, prf_hash_key, ope, tree_node.if_true_child, metaData)
        # Recurse to the if false tree_node
        enc_tree_node(he_pub_key, prf_hash_key, ope, tree_node.if_false_child, metaData)
    # else it is the bs.Leaf
    else:
        # Value....
        tree_node.value = paillier.encrypt(he_pub_key, tree_node.value)


def enc_xgboost_model(ppBoostKey: PPBoostKey, trees: list, metaData):
    """
    Encrypts the model (trees) to an encrypted format.
    :param ppBoostKey: the pp boost key wrapper.
    :param metaData: metaData containing min, max information
    :param trees: the model as an input (a list of trees)
    """
    he_pub_key = ppBoostKey.get_public_key()
    prf_hash_key = ppBoostKey.get_prf_key()
    ope = ppBoostKey.get_ope_encryptor()

    for t in trees:
        enc_tree_node(he_pub_key, prf_hash_key, ope, t, metaData)
    return trees


def predict_single_input_binary(trees, vector, default_base_score=0.5):
    """
    return a prediction on a single vector.
    :param trees: a list of trees (model represenation)
    :param vector: a single input vector
    :param default_base_score: a default score is 0.5 (global bias)
    :return: the predicted score
    """
    predict_sum_score = default_base_score
    for t in trees:
        score = t.eval(vector)
        predict_sum_score += score
    return predict_sum_score


def predict_binary(trees, vector, default_base_score=0.5):
    """
    Prediction on @vector over the @trees
    :param default_base_score: default score is 0.5 (according to the xgboost -- global bias)
    :param trees: list of trees
    :param vector: a list of input vectors
    :return: the prediction values (summation of the values from all the leafs)
    """
    result = []
    for index, row in vector.iterrows():
        # compute the score for all of the input vectors
        result.append(predict_single_input_binary(trees, row))
    # returns result as np array
    return np.array(result)


def predict_single_input_multiclass(trees, num_classes, vector):
    """
    return a prediction on a single input vector.
    The algorithm computes the sum of scores for all the corresponding classes (boosters in the xgboost model).
    For each class, it sum up all the leaves' values
    :param trees: a list of trees (model representation)
    :param num_classes: the total number classes to classify
    :param vector: a single input vector
    :return: the predicted score
    """
    num_trees = len(trees)

    # sum of score for each category: exp(score)
    result = []
    for i in range(num_classes):
        result.append(0)

    # this is to compute the softmax, however, server can only perform additvely homo operation,
    # so here we can compute scores seperately
    for i in range(num_trees):
        score = trees[i].eval(vector)

        result[i % num_classes] += score
    # return the result as a list (contains all of the scores for each labels).
    return result


def predict_multiclass(trees, num_classes, input_data_df):
    """
    This prediction for dataframe input. For each record,
    it calls the 'predict_single_input_multiclass'
    :param trees: models
    :param num_classes: number of the labels
    :param input_data_df: input dataframe
    :return: prediction with aggregated scores
    """
    predicts = []
    for index, row in input_data_df.iterrows():
        vect_results = predict_single_input_multiclass(trees, num_classes, row)
        predicts.append(vect_results)
    return predicts


def client_side_multiclass_compute(predictions):
    """
    Output the actual predictions using the softmax methods.
    In particular, it computes the normalized exponential function, which converts a vector
    of K real numbers and normalizes it into a probability distribution consisting of K
    probability scores proportional to the exponentials of the input numbers, i.e. SoftMax assigns
    decimal probabilities to each class in a multi-class problem.
    Those decimal probabilities must add up to 1.0.

    :param predictions: a list of predictions, where each element is a list that contains
        the scores for each inputs - here, we allow the client to receive a list of
        prediction (consistent to xgboost's prediction) for the corresponding queries.
    :return: predicted classes: a list of the most probable classes the model predicts.
    """
    final_output = []

    # Predictions is a vector of results, the value can be any real floating point number
    # e.g. predictions for k classes -- predictions = [[x_1,... x_k], [y_1,... y_k], [z_1, ..., z_k]]
    for predict_i in predictions:

        # gets the sum of all the exponential predicted results
        # e.g. e^(x_1) + ... + e^(x_k)
        sum_score = np.sum(np.exp(predict_i))
        output = []
        for x_i in predict_i:
            # for each t in predict_i
            # e.g. e^(x_i)/(e^(x_1) + ... + e^(x_k))
            output.append(np.exp(x_i) / sum_score)

        # report the argmax as the predicted class (most probable).
        final_output.append(np.argmax(output))
    return final_output


def client_decrypt_prediction_multiclass(private_key, predictions):
    """
    Client calls this function to compute the multi-class prediction -- i.e. it outputs the most probable class for the
    predicted class.
    This methods first decrypts the aggregated predictions using @private_key@,
    then calls the client_side_multiclass_compute() function.
    :param private_key: the private key to decrypt the values.
    :param predictions: encrypted list of predictions (each entry of the list is a list of encrypted scores.)
    :return: the results as numpy.ndarray()
    """
    decrypted_pred_list = []
    for enc_pred in predictions:
        decrypted_scores = []
        for enc_scores in enc_pred:
            # decrypts the encrypted scores.
            decrypted_scores.append(paillier.decrypt(private_key, enc_scores))
        decrypted_pred_list.append(decrypted_scores)
    result = client_side_multiclass_compute(decrypted_pred_list)
    return np.array(result)


# encrypts the input vector
def enc_input_vector(hash_key, ope, feature_set, input_vector, metadata):
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

    feature_list = list(feature_set)
    enc_feature_list = list()

    # Process feature list first.
    for i in feature_list:
        enc_feature_list.append(hmac_msg(hash_key, i))

    # calls the hmac_feature method to hmac the feature name of the input vector.
    hmac_feature(hash_key, input_vector)

    # Dropping the columns not in the feature list
    for col in input_vector.columns:
        if col not in enc_feature_list:
            input_vector.drop(col, axis=1, inplace=True)

    for i, row in input_vector.iterrows():
        # Encrypts all the features in the input_vector
        for feature in list(input_vector.columns.values):
            if not math.isnan(row[feature]):

                noramlized_feature = metadata.affine_transform(row[feature])

                if noramlized_feature > MAX_NUM_OPE_ENC or noramlized_feature < 0:
                    raise Exception("Invalid input: input is out of range (0, " + MAX_NUM_OPE_ENC +
                                    "). The system cannot encrypt",
                                    noramlized_feature)

                ope_value = ope.encrypt(int(noramlized_feature))

                input_vector.at[i, feature] = ope_value
