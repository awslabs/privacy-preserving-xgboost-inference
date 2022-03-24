# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from phe import PaillierPrivateKey, PaillierPublicKey
from ope.pyope.ope import OPE
from secrets import token_bytes
import ppxgboost.PaillierAPI as paillier


class PPModelKey:
    """
    Key used to encrypt a PPModel
    """

    def __init__(self, public_key: PaillierPublicKey, prf_key: bytes, ope_encryptor: OPE):
        """
        :param public_key: Paillier cryptosystem public key
        :param prf_key: PRF key
        :param ope_encryptor: OPE key
        """
        #  Private Attributes
        self.__public_key = public_key
        self.__prf_key = prf_key
        self.__ope_encryptor = ope_encryptor

    def get_public_key(self):
        """
        :return: the Paillier public key
        """
        return self.__public_key

    def get_prf_key(self):
        """
        :return: the PRF key
        """
        return self.__prf_key

    def get_ope_encryptor(self):
        """
        :return: the OPE encrypter
        """
        return self.__ope_encryptor


class PPQueryKey:
    """
    Key used to encrypt queries and decrypt query results
    """

    def __init__(self, private_key: PaillierPrivateKey, prf_key: bytes, ope_encryptor: OPE):
        """
        :param private_key: Paillier cryptosystem private key
        :param prf_key: PRF key
        :param ope_encryptor: OPE key
        """
        #  Private Attributes
        self.__private_key = private_key
        self.__prf_key = prf_key
        self.__ope_encryptor = ope_encryptor

    def get_private_key(self):
        """
        :return: the Paillier private key
        """
        return self.__private_key

    def get_prf_key(self):
        """
        :return: the PRF key
        """
        return self.__prf_key

    def get_ope_encryptor(self):
        """
        :return: the OPE encrypter
        """
        return self.__ope_encryptor


def generatePPXGBoostKeys():
    """
    Generate keys to encrypt an XGBoost model, encrypt queries, and decrypt query results.

    :return: model encryption keys, query encryption/decryption keys
    """

    # token bytes calls the os.urandom().
    prf_key = token_bytes(16)
    paillier_public_key, paillier_private_key = paillier.he_key_gen()
    ope_encryptor = OPE(token_bytes(16))
    return PPModelKey(paillier_public_key, prf_key, ope_encryptor), PPQueryKey(paillier_private_key, prf_key, ope_encryptor)
