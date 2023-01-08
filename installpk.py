import sys
import subprocess
import conda.cli.python_api as Conda

# implement conda as a subprocess:

subprocess.check_call('conda install -c conda-forge mlxtend')
subprocess.check_call('conda install -c conda-forge optuna')
subprocess.check_call('conda install -c conda-forge xgboost')
subprocess.check_call('conda install -c conda-forge lightgbm')
subprocess.check_call('conda install -c conda-forge scikit-learn')
subprocess.check_call('conda install -c conda-forge matplotlib')