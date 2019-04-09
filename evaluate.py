"""
This script is used for evaluating segmentation on input images
using a model whose name is given in arguments.

Arguments:
model_name: name of model to be used.
"""

import argparse
import os
import sys
import json
from data_loader import DataLoader
from input_fn import input_fn
from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan.")
parser.add_argument("-w", "--weight_file_name", help="Name of weight file name in model_weights directories. ")

try:
    with open("./params.json") as f:
        dataset_params = json.load(f)

except FileNotFoundError:
    print("params.json file doesn't exist.")
    exit()

if __name__ == "__main__":

    args = parser.parse_args()
    assert(args.model_name in ['unet', 'segan', 'attention-unet'])
    is_eager = True

    if args.model_name == 'segan':
        model_dir = './models/segan'
        sys.path.append(model_dir)
        from evaluation import evaluate
        from model import SegAN
        model = SegAN()
        weight_file_subpath = os.path.join('model_weights_' + dataset_params["test_datasets_folder"], args.weight_file_name)
    
    elif args.model_name == 'unet':
        model_dir = './models/unet'
        sys.path.append(model_dir)
        from evaluation import evaluate
        from base_model import Unet
        model = Unet()
        weight_file_subpath = os.path.join('model_weights_' + dataset_params["test_datasets_folder"], args.weight_file_name)

    elif args.model_name == 'attention-unet' :
        sys.path.append('./models/unet')
        model_dir = './models/unet/attention-unet'
        sys.path.append(model_dir)
        from evaluation import evaluate
        from model import AttentionalUnet
        model = AttentionalUnet()
        weight_file_subpath = os.path.join('attention-unet', 'model_weights_' + dataset_params["test_datasets_folder"], args.weight_file_name)

    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    params = Params(json_path)
    
    params.weight_file_subpath = weight_file_subpath

    print("SUB_WEIGHT_PATH", params.weight_file_subpath)

    data_loader = DataLoader(dataset_params)

    X_test, Y_test = data_loader.loadTestDatasets()

    test_inputs = input_fn(False, is_eager, X_test, Y_test, params)

    predict_model_specs = model.model_fn("eval", test_inputs)
    evaluate(predict_model_specs, params)
