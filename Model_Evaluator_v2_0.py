# Script for evaluating a number of NN-models in ASE

#  *** INITIAL SETUP *******************************

script_name = 'Model_Evaluator_v2_0.py'
print('\nInitializing ' + script_name +'\n')

# Standard imports:
import numpy as np
import pandas as pd
import os

# ASE imports:
import ase
from ase.io.trajectory import Trajectory

# NequIP imports:
import nequip
from nequip.ase import NequIPCalculator

# imports from Auxiliary file:
from Model_Evaluator_auxiliary_v1_0 import extract_all_data

# ***************************************************
# *** INPUT PARAMETERS AND FILE PATHS ***************

# Path and test data file name (ASE Trajectory file):
path_test_dataset = "AuTiO2.traj"

paths_models = [MODEL]

# paths_models = ["NequIP-layers=1.pth",
#                "NequIP-layers=2.pth",
#                "NequIP-layers=3.pth",
#                "NequIP-layers=4.pth",
#                "NequIP-layers=5.pth",
#                "NequIP-layers=6.pth",
#                "NequIP-layers=7.pth",
#                "NequIP-layers=8.pth",
#               ]

model_atomic_species = {
                        "Au": "Au",
                        "Ti": "Ti",
                        "O": "O"
                        }

output_folder = r'Evaluated_Models'

model_names = None

# ***************************************************
# *** IMPORTING MODEL CALCULATORS *******************

print('Retrieving models:', end = '')
calculators = []
for path in paths_models:
    
    # Initializing the model as a ASE calculator:
    new_calc = NequIPCalculator.from_deployed_model(
        model_path= path,
    species_to_type_name = model_atomic_species
    )
    calculators.append(new_calc)
print(' Done!')

# ***************************************************
# *** RETRIEVING TEST DATA PARAMETERS ***************

print('Loading and extracting test data:', end = '')
test_data = Trajectory(path_test_dataset) # Loading ASE Trajectory file
test_data = test_data[-1815:]
extracted_test_data = extract_all_data(test_data)
print(' Done!')

print('\n*** Extracting all data using models ****************')
i = 0
extracted_model_data = []
for cal in calculators:
    extracted_model_data.append(extract_all_data(test_data, cal))
    
    i += 1
    print(i,'/',len(calculators), ' models', sep = '')
print('Done!')
print('*****************************************************\n')

# ***************************************************
# *** COMPUTING ALL ERRORS **************************

# Test_data forces and energy:
f_x = extracted_test_data['f_x']
f_y = extracted_test_data['f_y']
f_z = extracted_test_data['f_z']
E = extracted_test_data['E']

print('Computing errors:', end = '')
for model_data in extracted_model_data:
    
    # Computing diffrence in force components:
    model_data['diff_f_x'] = model_data['f_x'] - f_x
    model_data['diff_f_y'] = model_data['f_y'] - f_y
    model_data['diff_f_z'] = model_data['f_z'] - f_z
    
    # Computing diffrence in force magnitude:
    model_data['diff_|f|'] = np.linalg.norm(
        model_data[['diff_f_x', 'diff_f_y', 'diff_f_z']]
    , axis = 1)
    
    # Computing the diffrence in energy:
    model_data['diff_E'] = model_data['E'] - E 
print('Done!')

# ***************************************************
# *** CREATING AND SAVING METADATA ******************

print('Creating metadata:', end = '')
for model_data in extracted_model_data:
    model_data['metadata'] = ''

    model_data.at[0, 'metadata'] = 'test'
    
print(' Done!')

# ***************************************************
# *** SAVING COMPUTATIONS ***************************

print('Saving:', end = '')

# Making output folder if it dose not exist:
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

#path = str(output_folder + '/ME_test_data.csv')
#extracted_test_data.to_csv(path, index = False)

for i in range(len(extracted_model_data)):
    
    model_data = extracted_model_data[i]
    
    if(model_names == None):
        path = str(output_folder + '/' 
                   + os.path.basename(paths_models[i][:-4])
                   + '.csv')
    else:
        model_names[i]
        
    model_data.to_csv(path, index = False)
print(' Done!')

print('\n--- Model evaluation completed successfully ---')