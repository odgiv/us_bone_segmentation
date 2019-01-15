import argparse
import logging
import os
import random
import sys
import numpy as np
from utils import Params, set_logger
from data_loader import DataLoader
from input_fn import input_fn

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan")

if __name__ == "__main__":

    args = parser.parse_args()
    model_dir = os.path.join('./models/', args.model_name)

    sys.path.append(model_dir)
    from model import model_fn 
    from training import train_and_evaluate

    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    params = Params(json_path)

    set_logger(os.path.join(model_dir, 'train.log'))
    logging.info("Loading the datasets...")

    data_loader = DataLoader()

    X_train, Y_train, X_val, Y_val = data_loader.loadTrainValDatasets()

    print("X_train shape {}".format(X_train.shape))
    print("Y_train shape {}".format(Y_train.shape))
    print("X_valid shape {}".format(X_val.shape))
    print("Y_valid shape {}".format(Y_val.shape))

    train_inputs = input_fn(True, X_train, Y_train, params)
    eval_inputs = input_fn(False, X_val, Y_val, params)

    train_model_specs = model_fn("train", train_inputs, params)
    eval_model_specs = model_fn("eval", eval_inputs, params, reuse=True)

    params.train_size = X_train.shape[0]
    params.eval_size = X_val.shape[0]

    train_and_evaluate(train_model_specs, eval_model_specs, params)