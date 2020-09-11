# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from phe import PaillierPrivateKey, PaillierPublicKey

from ope.pyope.ope import OPE


class PPBoostKey:
    """
    construct the PPBoost key for encrypting the model
    """

    def __init__(self, public_key: PaillierPublicKey, prf_key: bytes, ope_encryptor: OPE):
        """
        :param public_key: paillier public key
        :param prf_key: PRF key
        :param ope_encryptor: OPE key
        """
        self.__public_key = public_key
        self.__prf_key = prf_key
        self.__ope_encryptor = ope_encryptor

    def get_public_key(self):
        """
        :return: the paillier public key
        """
        return self.__public_key

    def get_prf_key(self):
        """
        :return: the prf key
        """
        return self.__prf_key

    def get_ope_encryptor(self):
        """
        :return: the ope encrypter
        """
        return self.__ope_encryptor


class ClientKey:
    """
    construct the client key (encrypting the data, decrypting the results)
    """

    def __init__(self, private_key: PaillierPrivateKey, prf_key: bytes, ope_encryptor: OPE):
        """
        :param private_key: private paillier key
        :param prf_key: prf key
        :param ope_encryptor: OPE key
        """
        #  Private Attributes
        self.__private_key = private_key
        self.__prf_key = prf_key
        self.__ope_encryptor = ope_encryptor

    def get_private_key(self):
        return self.__private_key

    def get_prf_key(self):
        """
        :return: the prf key
        """
        return self.__prf_key

    def get_ope_encryptor(self):
        """
        :return: the ope encrypter
        """
        return self.__ope_encryptor
