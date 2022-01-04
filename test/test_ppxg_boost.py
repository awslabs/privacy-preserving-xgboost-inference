# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# This test file mainly tests binary prediction for xgboost
# It tests all of the interfaces using OPE, Paillier, etc.
import sys
import pickle as pl
import pandas as pd
import random

from secrets import token_bytes
from ppxgboost import BoosterParser as boostparser
from ppxgboost import PPBooster as ppbooster
from ope.pyope.ope import OPE, ValueRange
from ppxgboost import PaillierAPI as paillier
from ppxgboost.PPBooster import MetaData

sys.path.append('../third-party')


# Testing class for the pytest. To run simply "pytest test/" this will run all of the test in the test directory.
class Test_PPMParser:

    # the testing for the parsing the model and the dumped trees.
    def test_model_parse(self):
        dir_path = "test_files/model_file.pkl"

        with open(dir_path, 'rb') as f:  # will close() when we leave this block
            testing_model = pl.load(f)

        # get the trees in string representation.
        # dump_tree is a list of strings, where each string is tree representation. See the docs in xgBoost for details.
        dump_tree = testing_model.get_dump()

        # use boostparser to convert the model (in strings) to tree objects.
        parsing_trees, feature_set, min_max = boostparser.model_to_trees(testing_model, {'min': 0, 'max': 1})

        # for each one of trees, test if the parsed tree is the same as the tree object (calling print in tree object)
        for i in range(len(parsing_trees)):
            assert dump_tree[i] == boostparser.tree_to_string(parsing_trees[i])

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
        # The folowing is to compute the scores for the decision tree on input vectors in plaintext!
        ################################################################################################
        feature_set = set()

        # As this test just to test the correctness of the encrypt_tree_node method
        # add min_max just to 'fake' some min-max value for this test
        tree = boostparser.parse_tree(t1, feature_set, min_max={'min': 0, 'max': 1})

        # The score list value in plaintext.
        score_value = list()
        # get each row indexing with input vector's head
        for index, row in input_vector.iterrows():
            score_value.append(tree.eval(row))

        ################################################################################################
        # The folowing is to compute the scores based on the OPE processed decision tree
        ################################################################################################

        # Set up encrytion materials.
        # token bytes calls the os.urandom().
        prf_key = token_bytes(16)
        OPE_key = token_bytes(16)
        encrypter = OPE(OPE_key)

        # create a copy of the input vector and plaintext trees
        test_input_vector = input_vector.copy()
        enc_tree = tree

        public_key, private_key = paillier.he_key_gen()

        # as this only test the enc_tree_node ope, add fake metadata (min and max) for this computation
        # just for testing purposes.
        metaDataMinMax = MetaData({'min': 0, 'max': 1000})

        # 1. Encrypts the input vector for prediction (using prf_key_hash and ope-encrypter) based on the feature set.
        ppbooster.enc_input_vector(prf_key, encrypter, feature_set, test_input_vector, metaDataMinMax)

        # 2. process the tree into ope_enc_tree
        ppbooster.enc_tree_node(public_key, prf_key, encrypter, enc_tree, metaDataMinMax)

        # 3. OPE evaluation based on OPE encrypted values in the tree nodes.
        encrypted_value = list()
        for index, row in test_input_vector.iterrows():
            score = enc_tree.eval(row)
            encrypted_value.append(score)

        dec_value = list()
        for c in encrypted_value:
            dec_value.append(paillier.decrypt(private_key, c))

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
