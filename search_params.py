import argparse
import os
from subprocess import check_call
import sys

import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/base_model', help='Directory containing params.json')
parser.add_argument('--input_dir', default='data/from_kaggle', help="Directory containing the dataset")

def launch_training_job(parent_dir, input_dir, train_file, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        input_dir: (string) directory containing the dataset
        train_file: (string) file for training 
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    train_file_noext = os.path.splitext(train_file)[0]
    model_dir = os.path.join(parent_dir, job_name, train_file_noext)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} {train} --model_dir={model_dir} --input_dir {input_dir}".format(python=PYTHON, train=train_file
                                                                                    , model_dir=model_dir
                                                                                    , input_dir=input_dir)
    print(cmd)
    check_call(cmd, shell=True)

if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # Model files
    files = ['daishu_solution.py', 'shiji_solution1.py']

    # Perform hypersearch over one parameter
    #learning_rates = [1e-4, 1e-3, 1e-2]
    augmentation_bools = ['False', 'True']
    
    for train_file in files:
      for augmentation in augmentation_bools:
          # Modify the relevant parameter in params
          #params.learning_rate = learning_rate
          params.augmentation = augmentation

          # Launch job (name has to be unique)
          #job_name = "learning_rate_{}".format(learning_rate)
          job_name = "augmentation_{}".format(augmentation)
          launch_training_job(args.parent_dir, args.input_dir, train_file, job_name, params)