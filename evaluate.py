"""
This script is used for evaluating segmentation on input images
using a model whose name is given in arguments.

Arguments:
model_name: name of model to be used.
"""

import argparse
import os
import sys
from data_loader import DataLoader
from input_fn import input_fn
from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan.")

if __name__ == "__main__":

    args = parser.parse_args()
    assert(args.model_name in ['unet', 'segan', 'attention-unet'])

    
    if args.model_name == 'segan':
        model_dir = './models/segan'
        sys.path.append(model_dir)
        from evaluation import evaluate    
        from model import SegAN
        model = SegAN()
    
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    
    params = Params(json_path)

    data_loader = DataLoader()
    X_test, Y_test = data_loader.loadTestDatasets()

    test_inputs = input_fn(False, X_test, Y_test, params)

    predict_model_specs = model.model_fn("eval", test_inputs, params)
    evaluate(predict_model_specs, params)
    


