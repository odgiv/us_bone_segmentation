import argparse
import logging
import os
import random
import sys
import shutil
import re
import json
import numpy as np
from utils import Params, set_logger, delete_dir_content
from data_loader import DataLoader
from input_fn import input_fn
import tensorflow as tf
from datetime import datetime
# from evaluate import evaluate

"""
python train.py --model_name unet
"""
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", required=True, help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan")
args = parser.parse_args()
assert(args.model_name in ['unet', 'segan', 'nested-unet', 'attentionUnet'])

try:
    with open("./params.json") as f:
        dataset_params = json.load(f)

except FileNotFoundError:
    print("params.json file doesn't exist.")
    exit()


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

elif args.model_name == 'attentionUnet':
    sys.path.append('./models/unet')
    model_dir = os.path.join('./models/unet/', args.model_name)
    sys.path.append(model_dir)
    from trainer_eager import train_and_evaluate
    from model import AttentionalUnet
    model = AttentionalUnet()


save_model_weights_dir = model_dir + '/model_weights/valid_img_vol_' + dataset_params["test_datasets_folder"] + '_' + datetime.now().strftime('%m-%d_%H-%M-%S') + '/'
if not os.path.isdir(save_model_weights_dir):
    os.makedirs(save_model_weights_dir)
else: 
    delete_dir_content(save_model_weights_dir)

set_logger(os.path.join(model_dir, 'train.log'))

if os.path.exists('./train_summaries'):
    shutil.rmtree('./train_summaries')
if os.path.exists('./eval_summaries'):
    shutil.rmtree('./eval_summaries')


logging.info("Loading the datasets...")

data_loader = DataLoader(dataset_params)

json_path = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
model_params = Params(json_path)
model_params.save_weights_path = save_model_weights_dir
model_params.model_name = args.model_name


# X_train, Y_train = data_loader.loadTrainDatasets()
# X_val, Y_val = data_loader.loadValidDatasets()
X_train = dataset_params.datasets_path + '/imgs'
Y_train = dataset_params.datasets_path + '/gts'
X_val = dataset_params.datasets_path + '/val_imgs'
Y_val = dataset_params.datasets_path + '/val_gts'
# print("X_train shape {}".format(X_train.shape))
# print("Y_train shape {}".format(Y_train.shape))
# print("X_valid shape {}".format(X_val.shape))
# print("Y_valid shape {}".format(Y_val.shape))

model_params.train_size = 3045 #X_train.shape[0]
model_params.eval_size = 79 #X_val.shape[0]
model_params.model_dir = model_dir

train_and_evaluate(model, X_train, Y_train, X_val, Y_val, model_params)
# train_and_evaluate(train_model_specs, val_model_specs, model_dir, model_params)
