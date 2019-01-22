import argparse
import logging
import os
import random
import sys
import numpy as np
from utils import Params, set_logger
from data_loader import DataLoader
from input_fn import input_fn
# from evaluate import evaluate

"""
python train.py --model_name unet
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan")

if __name__ == "__main__":

    args = parser.parse_args()
    assert(args.model_name in ['unet', 'segan', 'nested-unet', 'attention-unet'])    

    if args.model_name == 'segan':
        sys.path.append('./models/segan/')
        model_dir = './models/segan'
    elif args.model_name == 'unet':
        sys.path.append('./models/unet/')
        model_dir = './models/unet'
    else:
        sys.path.append('./models/unet/')
        model_dir = os.path.join('./models/unet/', args.model_name)

    sys.path.append(model_dir)
    
    from trainer import train_and_evaluate

    if args.model_name == "unet":
        from base_model import Unet
        model = Unet()

    elif args.model_name == "attention-unet":
        from model import AttentionalUnet
        model = AttentionalUnet()
        
    elif args.model_name == "nested-unet":
        # from model import UnetPlusPlus
        # model = UnetPlusPlus()
        pass
    elif args.model_name == "segan":
        # from model import SegAn
        # model = SegAn()    
        pass
        
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

    train_model_specs = model.model_fn("train", train_inputs, params)
    # sharing model weights for train and valid
    #eval_inputs["prediction"] = train_model_specs["prediction"]
    eval_inputs["model"] = train_model_specs["model"]

    eval_model_specs = model.model_fn("eval", eval_inputs, params, reuse=False)

    params.train_size = X_train.shape[0]
    params.eval_size = X_val.shape[0]

    train_and_evaluate(train_model_specs, eval_model_specs, model_dir, params)