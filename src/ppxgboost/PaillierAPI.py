# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import phe


# Wrapper of the pailler encryption scheme.
def he_key_gen(private_keyring=None, n_length=phe.paillier.DEFAULT_KEYSIZE):
    """
    Key generation of the paillier homomorphic encryption. The key_gen can take
    a private_key and n_length as parameters. The default key length is 2048
    :param private_keyring: this parameter is the private key that used to
        generate the public key. We usually do not set up this parameter.
    :param n_length: the key length -- the default key size is set to 2048.
    :return: public/private key pair
    """

    public_key, private_key = phe.generate_paillier_keypair(private_keyring, n_length)
    return public_key, private_key


def message_modulus_size(public_key):
    """
    :param public_key: the public key - pub_key:(n, g). In this case g = n+1
    :return: n -- the upper bound for the message space. The input message can
    be as large as n/3 based on the message encoding. Any number bigger than n/3
    will cause overflow.
    """
    # this message is defined in paillier message encoding to avoid overflow.
    # the input message (specified in the PHE library --> n/3) -> the message space
    # is not actually (0, n)
    return public_key.n // 3


def encrypt(public_key, msg, precision=None):
    """
    Encode and Paillier encrypt a real number *msg*.

        Args:
          msg: an int or float to be encrypted. If int, it must satisfy abs(*msg*) < public_key.n / 3.
            If float, it must satisfy abs(*msg* / *precision*) << n / 3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Passed to :method:`EncodedNumber.encode`.
            If *msg* is a float then *precision* is the maximum
            **absolute** error allowed when encoding *msg*. Defaults
            to encoding *msg* exactly.

    :param public_key: public key
    :param msg: message
    :param precision: this is optional, the maximum absolute error allowed when encoding *msg*
    :return: Enc(message)
    """
    return public_key.encrypt(msg, precision)


def decrypt(private_key, msg):
    """
    private_key to decrypt the message
    :param private_key: private key
    :param msg: encrypted number
    :return: decrypt and decode the number
    """
    return private_key.decrypt(msg)


def assert_ciphertext(encrypted_number):
    """
    Check if *encrypted_number* is a valid EncryptedNumber type
    :param encrypted_number: input
    exit if it is not.
    """
    if not isinstance(encrypted_number, phe.EncryptedNumber):
        raise ValueError("encrypted input is not a EncryptedNumber Type")


def ciphertext_serialization(public_key, encrypted_number):
    """
    EncryptedNumber:
    ciphertext (int) – encrypted representation of the encoded number.
    exponent (int) – used by EncodedNumber to keep track of fixed precision. Usually negative.
    Exponent will be zero when encrypting an integer.
    :param public_key: the public key
    :param encrypted_number: encrypted ciphertext
    :return:
    """
    encrypt_with_pub_key = {'public_key': {'n': public_key.n},
                            'values': [str(encrypted_number.ciphertext()), encrypted_number.exponent]}
    serialised = json.dumps(encrypt_with_pub_key)
    return serialised


def ciphertext_deserialization(serialised):
    """
    EncryptedNumber: ciphertext (int), exponent (int).
    :param serialised json format
    :return: public key and the ciphertext.
    """
    received_dict = json.loads(serialised)
    pk = received_dict['public_key']
    public_key_rec = phe.paillier.PaillierPublicKey(n=int(pk['n']))
    enc_nums_rec = phe.paillier.EncryptedNumber(public_key_rec, int(received_dict['values'][0]),
                                                int(received_dict['values'][1]))
    return public_key_rec, enc_nums_rec
