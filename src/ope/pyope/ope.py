import hmac
import math
import hashlib
import os
import base64

import six
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.modes import CTR

import ope.pyope.stat as stat
import ope.pyope.util as util

from ope.pyope.errors import InvalidCiphertextError, InvalidRangeLimitsError, OutOfRangeError

# See https://www.cc.gatech.edu/~aboldyre/papers/bclo.pdf for the security advantage of
# of the adversary. In_Range is M, and Out_Range is N.
# Unlike the typical CPA security from original, the OPE security (popf-cca)
# The advantage of the popf-cca (pseudorandom order-preserving function advantage
# under chosen-ciphertext attack) adv[popf-cca] <= adv[lf-prf] + epsilon if the adversary
# to the popf-cca makes at most q oracle accesses.
# The adv[lf-prf] is approximately 1/(M choose N) when making at most 5log N + 1 queries.


# The input interval is [0, 2^43 - 1]
DEFAULT_IN_RANGE_START = 0
DEFAULT_IN_RANGE_END = 2 ** 43 - 1

# The output interval is [0, 2^63 - 1]
DEFAULT_OUT_RANGE_START = 0
DEFAULT_OUT_RANGE_END = 2 ** 63 - 1


# The advantage below is 1/(2^63 choose 2^43) compared to the previous parameter
# setup 1/(2^63 choose 2^32) to break the the length-flexible PRF thus the popf-cca

class ValueRange(object):
    """A range of consecutive integers with the specified boundaries (both inclusive)"""

    def __init__(self, start, end):
        if type(start) not in six.integer_types:
            raise InvalidRangeLimitsError("Invalid range start: must be integer")

        if type(end) not in six.integer_types:
            raise InvalidRangeLimitsError("Invalid range end: must be integer")

        if start > end:
            raise InvalidRangeLimitsError("Invalid range: the start of the range is greater than the end")

        self._start = start
        self._end = end

    def __repr__(self):
        return 'ValueRange({0.start!r}, {0.end!r})'.format(self)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if type(value) not in six.integer_types:
            raise ValueError('Start value must be integer')
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        if type(value) not in six.integer_types:
            raise ValueError('End value must be integer')
        self._end = value

    def size(self):
        """Return the range length, including its start and end"""
        return self.end - self.start + 1

    def range_bit_size(self):
        """Return a number of bits required to encode any value within the range"""
        return int(math.ceil(math.log(self.size(), 2)))

    def contains(self, number):
        """Check if the number is within the range"""
        return self.start <= number <= self.end

    def copy(self):
        """Make a copy of the range"""
        return ValueRange(self.start, self.end)


class OPE(object):

    def __init__(self, key, in_range=None, out_range=None):
        if not isinstance(key, bytes):
            raise TypeError("key: expected bytes, but got %r" % type(key).__name__)
        self.key = key

        if in_range is None:
            in_range = ValueRange(DEFAULT_IN_RANGE_START, DEFAULT_IN_RANGE_END)
        self.in_range = in_range

        if out_range is None:
            out_range = ValueRange(DEFAULT_OUT_RANGE_START, DEFAULT_OUT_RANGE_END)
        self.out_range = out_range

        if in_range.size() > out_range.size():
            raise Exception('Invalid range')

    def encrypt(self, plaintext):
        """Encrypt the given plaintext value"""
        if not isinstance(plaintext, int):
            raise ValueError('Plaintext must be an integer value')
        if not self.in_range.contains(plaintext):
            raise OutOfRangeError('Plaintext is not within the input range')
        return self.encrypt_recursive(plaintext, self.in_range, self.out_range)

    '''
     in_range is the Domain (M in the paper)
     out_range is the Domain (N in the paper)
     The bijective function f(x):[M] -> [N]
    '''

    def encrypt_recursive(self, plaintext, in_range, out_range):
        in_size = in_range.size()  # M
        out_size = out_range.size()  # N
        in_edge = in_range.start - 1  # d
        out_edge = out_range.start - 1  # r
        mid = out_edge + int(math.ceil(out_size / 2.0))  # y
        assert in_size <= out_size
        if in_range.size() == 1:
            coins = self.tape_gen(plaintext)
            ciphertext = stat.sample_uniform(out_range, coins)
            return ciphertext
        coins = self.tape_gen(mid)
        x = stat.sample_hgd(in_range, out_range, mid, coins)

        if plaintext <= x:
            in_range = ValueRange(in_edge + 1, x)
            out_range = ValueRange(out_edge + 1, mid)
        else:
            in_range = ValueRange(x + 1, in_edge + in_size)
            out_range = ValueRange(mid + 1, out_edge + out_size)
        return self.encrypt_recursive(plaintext, in_range, out_range)

    def decrypt(self, ciphertext):
        """Decrypt the given ciphertext value"""
        if not isinstance(ciphertext, int):
            raise ValueError('Ciphertext must be an integer value')
        if not self.out_range.contains(ciphertext):
            raise OutOfRangeError('Ciphertext is not within the output range')
        return self.decrypt_recursive(ciphertext, self.in_range, self.out_range)

    def decrypt_recursive(self, ciphertext, in_range, out_range):
        in_size = in_range.size()  # M
        out_size = out_range.size()  # N
        in_edge = in_range.start - 1  # d
        out_edge = out_range.start - 1  # r
        mid = out_edge + int(math.ceil(out_size / 2.0))  # y
        assert in_size <= out_size
        if in_range.size() == 1:
            in_range_min = in_range.start
            coins = self.tape_gen(in_range_min)
            sampled_ciphertext = stat.sample_uniform(out_range, coins)
            if sampled_ciphertext == ciphertext:
                return in_range_min
            else:
                raise InvalidCiphertextError('Invalid ciphertext')
        coins = self.tape_gen(mid)
        x = stat.sample_hgd(in_range, out_range, mid, coins)

        if ciphertext <= mid:
            in_range = ValueRange(in_edge + 1, x)
            out_range = ValueRange(out_edge + 1, mid)
        else:
            in_range = ValueRange(x + 1, in_edge + in_size)
            out_range = ValueRange(mid + 1, out_edge + out_size)
        return self.decrypt_recursive(ciphertext, in_range, out_range)

    def tape_gen(self, data):
        """Return a bit string, generated from the given data string"""

        # FIXME
        data = str(data).encode()

        # Derive a key from data
        hmac_obj = hmac.HMAC(self.key, digestmod=hashlib.sha256)
        hmac_obj.update(data)
        assert hmac_obj.digest_size == 32
        digest = hmac_obj.digest()

        # Use AES in the CTR mode to generate a pseudo-random bit string
        aes_algo = algorithms.AES(digest)
        aes_cipher = Cipher(aes_algo, mode=CTR(b'\x00' * 16), backend=default_backend())
        encryptor = aes_cipher.encryptor()

        while True:
            encrypted_bytes = encryptor.update(b'\x00' * 16)
            # Convert the data to a list of bits
            bits = util.str_to_bitstring(encrypted_bytes)
            for bit in bits:
                yield bit

    @staticmethod
    def generate_key(block_size=32):
        """Generate random key for ope cipher.

        Parameters
        ----------
        block_size : int, optional
            Length of random bytes.

        Returns
        -------
        random_key : str
            A random key for encryption.

        Notes:
        ------
        Implementation follows https://github.com/pyca/cryptography
        """
        random_seq = os.urandom(block_size)
        random_key = base64.b64encode(random_seq)
        return random_key
