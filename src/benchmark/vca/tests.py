import tensorflow as tf
import numpy as np
import random
import time
import json
import os
import argparse
from math import sqrt

try:
    from config import config
    from DilatedRNN import DilatedRNNWavefunction
    from utils import Fullyconnected_localenergies, Fullyconnected_diagonal_matrixelements
    from vca import vca_solver
    print("Successfully imported local modules.")
except ImportError:
    print("Local .py files not found. Please ensure they are in the same directory or paste the classes below.")

tf.compat.v1.disable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument("test_dir")
parser.add_argument("result_file")
args = parser.parse_args()

test_dir = args.test_dir
result_file = args.result_file


path = os.path.split(test_dir)[0]
test_files = [os.path.join(path, file) for file in os.listdir(test_dir)]
# Filter test files for seed numbers 1-25
filtered_files = []
for file in test_files:
    try:
        int(file[-6:-4])
    except ValueError:
        filtered_files.append(file)
        continue
    if int(file[-6:-4]) <= 25:
        filtered_files.append(file)

filtered_files = sorted(filtered_files)
print(filtered_files)
result_file = result_file
if os.path.isfile(result_file):
    with open(result_file, 'r') as f:
        data = json.load(f)
        minimums = data['results']
else:
    minimums = dict()
print(minimums)
for test in filtered_files:
    # 1. Path to your problem instance
    if os.path.split(test)[1] in minimums.keys():
        print(test + " previously completed")
        continue
    seed = 0
    print(test)

    # 2. Initialize configuration
    vca_config = config(test, seed)

    # 3. Run the solver
    # This will output the annealing progress, energy (E), and free energy (F)
    mean_energies, min_energies = vca_solver(vca_config)

    print(f"Minimum Energy Found: {min_energies}")
    minimums[os.path.split(test)[1]] = min_energies
    with open(result_file, 'w') as f:
        json.dump({'tests-completed': len(minimums), 'results': minimums}, f, indent=4)

print(minimums)

