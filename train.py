import argparse
import logging
import os
import random
import sys
import shutil
import re
import numpy as np
from utils import Params, set_logger
from data_loader import DataLoader
from input_fn import input_fn
# from evaluate import evaluate

"""
python train.py --model_name unet
"""

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan")

if __name__ == "__main__":

    args = parser.parse_args()
    assert(args.model_name in ['unet', 'segan',
                               'nested-unet', 'attention-unet'])
    
    is_eager = True

    if args.model_name == 'segan':
        model_dir = './models/segan'
        sys.path.append(model_dir)
        from trainer_eager import train_and_evaluate
        from model import SegAN
        model = SegAN()

    elif args.model_name == 'unet':
        model_dir = './models/unet'
        sys.path.append(model_dir)
        from trainer_eager import train_and_evaluate
        from base_model import Unet
        model = Unet()
        is_eager = True


    elif args.model_name == 'attention-unet':
        sys.path.append('./models/unet')
        model_dir = os.path.join('./models/unet/', args.model_name)
        sys.path.append(model_dir)
        from trainer_eager import train_and_evaluate
        from model import AttentionalUnet
        model = AttentionalUnet()
        is_eager = True

    else:
        print("No model named " + args.model_name + " exists.")
        exit(0)

    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    params = Params(json_path)

    set_logger(os.path.join(model_dir, 'train.log'))

    if os.path.exists('./train_summaries'):
        shutil.rmtree('./train_summaries')
    if os.path.exists('./eval_summaries'):
        shutil.rmtree('./eval_summaries')

    logging.info("Loading the datasets...")

    data_loader = DataLoader()

    X_train, Y_train, X_val, Y_val = data_loader.loadTrainValDatasets()

    print("X_train shape {}".format(X_train.shape))
    print("Y_train shape {}".format(Y_train.shape))
    print("X_valid shape {}".format(X_val.shape))
    print("Y_valid shape {}".format(Y_val.shape))

    train_inputs = input_fn(True, is_eager, X_train, Y_train, params)
    val_inputs = input_fn(True, is_eager, X_val, Y_val, params)

    train_model_specs = model.model_fn("train", train_inputs, params)

    # sharing model weights for train and valid
    #eval_inputs["prediction"] = train_model_specs["prediction"]
    # eval_inputs["model"] = train_model_specs["model"]

    # val_model_specs = model.model_fn("eval", val_inputs, params, reuse=True)
    val_model_specs = val_inputs

    params.train_size = X_train.shape[0]
    params.eval_size = X_val.shape[0]

    train_and_evaluate(train_model_specs, val_model_specs, model_dir, params)
