# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import ppxgboost.PaillierAPI as paillier
from ppxgboost.PPModel import *
from ppxgboost.PPQuery import PPQuery


def predict_single_input_binary(model: PPModel, query, default_base_score=0.5):
    """
    Return the score for this query in a binary classification model

    :param model: the model to evaluate
    :param query: a single input query
    :param default_base_score: a default score is 0.5 (global bias)
    :return: the query's score
    """

    scores = model.eval(query)
    return default_base_score + sum(scores)

def predict_binary(enc_model: PPModel, enc_queries: [PPQuery], default_base_score=0.5):
    """
    Prediction on @enc_queries over the @enc_model
    :param enc_model: encrypted PPModel
    :param enc_queries: a list of encrypted PPQuery
    :param default_base_score: default score is 0.5 (according to the xgboost -- global bias)
    :return: a list of prediction values
    """

    return list(map(lambda q: predict_single_input_binary(enc_model, q, default_base_score), enc_queries))


def predict_single_input_multiclass(model: PPModel, num_classes, query):
    """
    Return a prediction on a single input vector.
    The algorithm computes the sum of scores for all the corresponding classes (boosters in the xgboost model).
    For each class, it sum up all the leaves' values

    :param model: internal model representation
    :param num_classes: the total number classes to classify
    :param query: a single input query
    :return: the predicted score
    """
    scores = model.eval(query)

    # sum of score for each category: exp(score)
    result = []
    for i in range(num_classes):
        result.append(0)

    # this is to compute the softmax, however, server can only perform
    # additvely homomorphic operation, so here we can compute scores
    # seperately
    for i in range(len(scores)):
        j = i % num_classes
        result[j] = result[j] + scores[i]

    return result

def predict_multiclass(enc_model: PPModel, num_classes, enc_queries: [PPQuery]):
    """
    This prediction for dataframe input. For each record,
    it calls the 'predict_single_input_multiclass'
    :param enc_model: encrypted PPModel
    :param num_classes: the total number classes to classify
    :param enc_queries: list of encrypted PPQuery
    :return: a list of prediction with aggregated scores
    """

    return list(map(lambda q: predict_single_input_multiclass(enc_model, num_classes, q), enc_queries))

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
