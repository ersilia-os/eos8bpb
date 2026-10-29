# imports
import sys
import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out

from predict import COLUMNS, my_model

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = my_model(smiles_list)

# check input and output have the same length
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
write_out(outputs, COLUMNS, output_file, np.float32)
