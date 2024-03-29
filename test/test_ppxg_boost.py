# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# This test file mainly tests binary prediction for xgboost
# It tests all of the interfaces using OPE, Paillier, etc.
import pickle as pl
import pandas as pd
import random

import pyope.ope as pyope
from ppxgboost import PaillierAPI as paillier
import ppxgboost.OPEMetadata as OPEMetadata
import ppxgboost.PPModel as PPModel
import ppxgboost.PPTree as PPTree
import ppxgboost.PPKey as PPKey
import ppxgboost.PPQuery as PPQuery

# The tests require modified input and output ranges
in_range = pyope.ValueRange(pyope.DEFAULT_IN_RANGE_START, 2 ** 43 - 1)
out_range = pyope.ValueRange(pyope.DEFAULT_OUT_RANGE_START, 2 ** 63 - 1)

# Testing class for the pytest. To run simply "pytest test/" this will run all of the test in the test directory.
class Test_PPMParser:

    # the testing for the parsing the model and the dumped trees.
    def test_model_parse(self):
        dir_path = "test_files/model_file.pkl"

        with open(dir_path, 'rb') as f:  # will close() when we leave this block
            testing_model = pl.load(f)

        ppmodel = PPModel.from_xgboost_model(testing_model)

        # get the trees in string representation.
        # dump_tree is a list of strings, where each string is tree representation. See the docs in xgBoost for details.
        dump_tree = testing_model.get_dump()

        # for each one of trees, test if the parsed tree is the same as the tree object (calling print in tree object)
        for i in range(len(dump_tree)):
            assert dump_tree[i] == PPTree.tree_to_string(ppmodel.trees[i])

    def test_ope_node(self):
        """
        Test vector on evaluating the decision tree based on OPE
        :return:
        """
        t1 = '0:[Pclass<3] yes=1,no=2,missing=1\n\t1:[Fare<13] yes=3,no=4,missing=3\n\t\t3:leaf=323\n\t\t4:[' \
             'Age<42] yes=9,no=10,missing=10\n\t\t\t9:leaf=32434\n\t\t\t10:leaf=43124\n\t2:[Age<6] yes=5,' \
             'no=6,missing=6\n\t\t5:[SibSp<32] yes=11,no=12,' \
             'missing=11\n\t\t\t11:leaf=9473\n\t\t\t12:leaf=836\n\t\t6:leaf=46\n '

        input_vector = pd.read_csv('test_files/test_prediction_input.csv')

        ################################################################################################
        # The following is to compute the scores for the decision tree on input vectors in plaintext!
        ################################################################################################

        # As this test just to test the correctness of the encrypt_tree_node method
        ppModel = PPModel.PPModel([PPTree.parse_tree(t1)])
        features = ppModel.get_features()

        # The score list value in plaintext.
        score_value = list()
        # get each row indexing with input vector's head
        queries = PPQuery.pandas_to_queries(input_vector)
        # Evaluating a model produces a list of results; we only want the result for the (one and) only tree in this model
        score_value = list(map(lambda q: ppModel.eval(q)[0], queries))

        ################################################################################################
        # The following is to compute the scores based on the OPE processed decision tree
        ################################################################################################

        # Set up encrytion materials.
        ppModelKey, ppQueryKey = PPKey.generatePPXGBoostKeys(in_range, out_range)

        # as this only test the enc_tree_node ope, add fake metadata (min and max) for this computation
        # just for testing purposes.
        metadata = OPEMetadata.OPEMetadata(ppModel, 0, 1000, in_range.end)

        # 1. Encrypts the input vector for prediction (using prf_key_hash and ope-encrypter) based on the feature set.

        queryEncryptor = PPQuery.QueryEncryptor(ppQueryKey, features, metadata)
        enc_queries = PPQuery.encrypt_queries(queryEncryptor, queries)

        # 2. process the tree into ope_enc_tree
        enc_model = ppModel.encrypt(ppModelKey, metadata)

        # 3. OPE evaluation based on OPE encrypted values in the tree nodes.
        # Evaluating a model produces a list of results; we only want the result for the (one and) only tree in this model
        encrypted_value = list(map(lambda q: enc_model.eval(q)[0], enc_queries))

        dec_value = list()
        for c in encrypted_value:
            dec_value.append(paillier.decrypt(ppQueryKey.get_private_key(), c))

        # 4. compare
        assert dec_value == score_value

    def test_paillier_encryption(self):

        # randomly generate a number
        a = random.getrandbits(64)

        # Key Generation
        pub_key, priv_key = paillier.he_key_gen()
        # encrypt the random number
        enc_a = paillier.encrypt(pub_key, a)

        # test if decryption works.
        assert a == paillier.decrypt(priv_key, enc_a)

    def test_paillier_api(self):

        pub_key, priv_key = paillier.he_key_gen()

        # randomly generate the a and b from [0, sixty_four_bit_num)
        a = random.getrandbits(64)
        b = random.getrandbits(64)

        # encrypt a & b
        enc_a = paillier.encrypt(pub_key, a)
        enc_b = paillier.encrypt(pub_key, b)

        # homomorphically evaluate a + b
        # enc_a_b = enc(a+b)
        paillier.assert_ciphertext(enc_a)
        paillier.assert_ciphertext(enc_b)
        enc_a_b = enc_a + enc_b

        # try to catch the exception if the input is NOT a ciphertext
        try:
            paillier.assert_ciphertext(a)
        except ValueError:
            print("Input was not a ciphertext")
            assert True

        scalar = random.getrandbits(64)

        # encrypted_c_scalar = enc(scalar * a)
        encrypted_c_scalar = scalar * enc_a

        # perform all decryptions
        dec_a_b = paillier.decrypt(priv_key, enc_a_b)
        dec_c_scalar = paillier.decrypt(priv_key, encrypted_c_scalar)

        assert dec_a_b == a + b
        assert dec_c_scalar == scalar * a

    def test_paillier_negative_num_test(self):

        pub_key, priv_key = paillier.he_key_gen()

        # Negative number test #
        x = random.randint((-1) * random.getrandbits(64), 0)

        # encrypt negative number x
        enc_x = paillier.encrypt(pub_key, x)
        # encrypted negative number -x (a positive num)
        enc_pos_x = paillier.encrypt(pub_key, (-1) * x)

        paillier.assert_ciphertext(enc_pos_x)
        paillier.assert_ciphertext(enc_x)
        # homomorphic addition (enc_sum <- enc(0))
        enc_sum = enc_pos_x + enc_x

        dec_x = paillier.decrypt(priv_key, enc_x)
        dec_sum = paillier.decrypt(priv_key, enc_sum)
        assert dec_x == x
        assert dec_sum == 0

    def test_serialization(self):

        pub_key, priv_key = paillier.he_key_gen()

        # randomly generate 64 bit number
        message = random.getrandbits(64)

        # Encrypt a message
        encrypted_message = paillier.encrypt(pub_key, message)
        # serialize it to json format
        json_str = paillier.ciphertext_serialization(pub_key, encrypted_message)

        # deserialize the json format
        pub_key, encrypted_number = paillier.ciphertext_deserialization(json_str)
        # decrypted the message using the correct private key.
        check_number = paillier.decrypt(priv_key, encrypted_number)

        assert message == check_number
