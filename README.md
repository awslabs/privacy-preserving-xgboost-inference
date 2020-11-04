# Privacy-Preserving XGBoost Inference

## Description
This repo is a prototype implementation of privacy-preserving xgBoost (https://xgboost.readthedocs.io/en/latest/).
We adopt several property-preserving encryption schemes to encrypt the xgBoost model so that
the privacy-preserving model can predict an encrypted data. An extended abstract of this work appears in Privacy-preserving Machine Learning Workshop at NeurIPS 2020.

## Development

*pip install pytest*

When run the test files, first in the repo directory.

- pip3 install -r requirements.txt

Go to the test directory ('cd test'), run the following:
- python -m pytest


The OPE scheme is from the open source (https://github.com/tonyo/pyope).
 It implements the OPE scheme by Boldyreva et. al. The source code is place in the 'third-party/ope/' directory.
 We also leverage the partially homomorphic encryption scheme
 (Paillier Cryptosystem: https://en.wikipedia.org/wiki/Paillier_cryptosystem), run __pip install phe__ to
 install this.

See [DEVELOPMENT.md](./DEVELOPMENT.md)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
