# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This test file tests the Key wrapper (PPKey.py)
from random import randrange
from secrets import token_bytes
from ppxgboost import PaillierAPI as paillier
from ppxgboost.PPKey import ClientKey
from ppxgboost.PPKey import PPBoostKey
import pyope.ope as pyope

# The tests require modified input and output ranges
in_range = pyope.ValueRange(pyope.DEFAULT_IN_RANGE_START, 2 ** 43 - 1)
out_range = pyope.ValueRange(pyope.DEFAULT_OUT_RANGE_START, 2 ** 63 - 1)

class Test_Key:
    def test_get_PPBoost_key(self):
        """
        Testing the PPBoost Key Wrapper.
        """

        # Build the PPBoostKey
        prf_key = token_bytes(16)
        ope_encrypter = pyope.OPE(token_bytes(16), in_range, out_range)
        public_key, private_key = paillier.he_key_gen()
        ppBoostKey = PPBoostKey(public_key, prf_key, ope_encrypter)

        a = randrange(pow(2, 30))
        b = randrange(pow(2, 30))

        # test the ope key
        ea = ppBoostKey.get_ope_encryptor().encrypt(a)
        eb = ppBoostKey.get_ope_encryptor().encrypt(b)
        assert (a < b) == (ea < eb)

        ea = ppBoostKey.get_public_key().encrypt(a)
        eb = ppBoostKey.get_public_key().encrypt(b)
        assert private_key.decrypt(ea + eb) == a + b

    def test_get_private_key(self):
        """
        Testing the Client Key Wrapper.
        """

        # Build the ClientKey
        prf_key = token_bytes(16)
        ope_encrypter = pyope.OPE(token_bytes(16), in_range, out_range)
        public_key, private_key = paillier.he_key_gen()
        clientKey = ClientKey(private_key, prf_key, ope_encrypter)

        a = randrange(pow(2, 30))
        b = randrange(pow(2, 30))

        # test the ope key
        ea = clientKey.get_ope_encryptor().encrypt(a)
        eb = clientKey.get_ope_encryptor().encrypt(b)
        assert (a < b) == (ea < eb)

        ea = public_key.encrypt(a)
        eb = public_key.encrypt(b)
        assert clientKey.get_private_key().decrypt(ea + eb) == a + b
