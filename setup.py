from setuptools import setup, find_packages

setup(name="ppxgboost",
      version='0.0.1',
      package_dir={'': 'src'},
      install_requires=['phe', 'pandas', 'xgboost', 'numpy', 'flask', 'scikit-learn', 'pytest'])
