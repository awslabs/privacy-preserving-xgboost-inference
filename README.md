# Privacy-Preserving XGBoost Inference

## Description
This repo is a prototype implementation of privacy-preserving XGBoost (https://xgboost.readthedocs.io/en/latest/).
We adopt several property-preserving encryption schemes to encrypt the XGBoost model so that
the privacy-preserving model can predict an encrypted query. 

An extended abstract of this work (https://arxiv.org/abs/2011.04789) appears in Privacy-preserving Machine Learning Workshop at NeurIPS 2020.

## Development

This package requires python>=3.8. Install the dependencies with

 - xargs -L 1 pip install < requirements.txt

This command installs the dependencies in a specific order.

Run the tests with:
- cd test
- python -m pytest

This package depends on the Paillier partially homomorphic encryption scheme (https://en.wikipedia.org/wiki/Paillier_cryptosystem). It also includes source code for a modified version of Boldyreva et. al.'s order-preserving encryption scheme (https://github.com/tonyo/pyope). The source code is place in the 'third-party/ope/' directory.

See [DEVELOPMENT.md](./DEVELOPMENT.md)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
