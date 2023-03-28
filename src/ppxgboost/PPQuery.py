# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import pandas as pd
import base64
import encodings
import hashlib
import hmac

from ppxgboost.OPEMetadata import *
from ppxgboost.PPKey import PPQueryKey

def hmac_msg(prf_key_hash: bytes, feature):
    """
    HMAC of a string using the provided key

    :param prf_key_hash: hash key as bytes
    :param feature: feature name as a string (encoded using 'UTF-8')
    :return: HMAC of feature name
    """

    # The encryption scheme requires a PRF to create 'pseudonyms' for the features.
    # We instantiate the PRF with HMAC using utf-8 encoding (python3) for feature names.
    # A reference that shows HMAC is a PRF (see Theorem 1 in https://eprint.iacr.org/2014/578.pdf)
    message = bytes(feature, encodings.utf_8.getregentry().name)
    sig = base64.b64encode(hmac.new(prf_key_hash, message, hashlib.sha256).digest())
    return sig.decode()

class PPQuery:
    """
    PPXGBoost query object
    """

    def __init__(self, query_dict):
        """
        :param query_dict: A dictionary containing the query labels and values
        """
        self.query_dict = query_dict

class QueryEncryptor:
    """
    Encryptor object for queries
    """

    def __init__(self, query_key: PPQueryKey, feature_set, metadata: OPEMetadata):
        """
        :param query_key: The query encryption/decryption key
        :param feature_set: The set of model features to encrypt in a query
        :param metadata: OPE metadata
        """

        self.key = query_key
        self.feature_set = feature_set
        self.metadata = metadata
        self.feature_name_map = {}
        for f in self.feature_set:
            self.feature_name_map[f] = hmac_msg(self.key.get_prf_key(), f)

    def encrypt_query(self, query: PPQuery):
        """
        Encrypt a single query by HMACing each feature name and OPE-encrypting
        each value.

        :param query: An unencrypted PPQuery
        :return: An encrypted PPQuery
        """

        encrypted_query = {}
        for k, v in query.query_dict.items():
            if k not in self.feature_set:
                continue

            # TODO: datasets in the tests *do* contain NaN
            # but this leaks where NaNs are...
            if math.isnan(v):
                encrypted_val = v
            else:
                normalized_val = self.metadata.affine_transform(v)

                if normalized_val > self.metadata.max_num_ope_enc or normalized_val < 0:
                    raise Exception("Invalid input: input is out of range (0, " + str(self.metadata.max_num_ope_enc) +
                                    "). The system cannot encrypt", normalized_val)
                encrypted_val = self.key.get_ope_encryptor().encrypt(int(normalized_val))
            encrypted_query[self.feature_name_map[k]] = encrypted_val
        return PPQuery(encrypted_query)

def encrypt_queries(encryptor: QueryEncryptor, queries: [PPQuery]):
    return list(map(lambda q: encryptor.encrypt_query(q), queries))

def pandas_to_queries(data_set: pd.DataFrame):
    """
    Convert data parsed with pandas into a list of queries

    :param data_set: A pandas data frame
    :return: list of queries
    """

    queries = []
    for i, row in data_set.iterrows():
        query = {}
        for feature in list(data_set.columns.values):
            query[feature] = row[feature]
        queries.append(PPQuery(query))
    return queries

# pytest will try to do weird things if you have a function name that starts with `test_`!!
# "fixture 'test_data' not found"
def get_test_data_extreme_values(queries: [PPQuery]):
    """
    Extract the minimum and maximum values from a list of queries.

    The `OPEMetadata` class requires an estimate for the minimum and maximum
    values across all features and all queries. For tests and examples,
    the entire test dataset is known in advance, so we can just extract
    the minimum and maximum values from the dataset directly.

    :param test_data: A non-empty list of queries
    :return: minimum value across all queries, maximum value across all queries
    """

    def query_extremes(q: PPQuery):
        values = q.query_dict.values()
        return min(values), max(values)

    extremes = map(query_extremes, queries)
    # unpack the list of tuples into mins and maxes
    q_mins, q_maxs = list(zip(*extremes))

    return min(q_mins), max(q_maxs)
