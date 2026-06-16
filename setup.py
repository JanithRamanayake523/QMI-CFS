"""Setup script for QMI-CFS."""
from setuptools import setup, find_packages

setup(
    name='qmi-cfs',
    version='0.1.0',
    description='Quantum Mutual Information-Based Feature Selection for Treatment Effect Estimation',
    author='Research Team',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'statsmodels>=0.14.0',
        'BorutaPy>=0.3.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'pytest-cov', 'black', 'flake8'],
    },
)
