from setuptools import setup, find_packages

setup(name="ppxgboost",
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      install_requires=['phe', 'pandas', 'xgboost', 'numpy', 'flask', 'scikit-learn', 'pytest'])
