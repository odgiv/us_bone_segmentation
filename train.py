import argparse
import logging
import os
import random
import sys
import shutil
import re
import json
import numpy as np
import tensorflow as tf
from utils import Params, set_logger, delete_dir_content, augmented_img_and_mask_generator, img_and_mask_generator
from data_loader import DataLoader
from input_fn import input_fn
from tqdm import tqdm
from datetime import datetime

opts = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
config = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config)
tfe = tf.contrib.eager

"""
python train.py --model_name unet
"""
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", required=True, help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan")
parser.add_argument("-d", "--datasets_dir", required=True, help="Path to a parent folder where datasets exist")
parser.add_argument("-l", "--learning_rate", type=float, default=0.0003)
parser.add_argument("-ld", "--learning_rate_decay", type=float, default=0.0)
parser.add_argument("-b", "--batch_size", type=int, default=10)
parser.add_argument("-n", "--num_epochs", type=int, default=300)
parser.add_argument("-s", "--save_summary_steps", type=int, default=100)
parser.add_argument("-id", "--experiment_id", type=int, required=True)

args = parser.parse_args()
assert(args.model_name in ['unet', 'segan', 'nested-unet', 'attentionUnet'])
assert(os.path.exists(args.datasets_dir))

print("tf version: ",  tf.__version__)

if args.model_name == 'segan':
    model_dir = './models/segan'
    sys.path.append(model_dir)
    from trainer_eager import train_step, evaluate
    from model import SegAN
    net = SegAN()
    segmentor_net = net.segNet
    critic_net = net.criNet
    
elif args.model_name == 'unet':
    model_dir = './models/unet'
    sys.path.append(model_dir)
    from trainer_eager import train_step, evaluate
    from base_model import Unet
    segmentor_net = Unet()

elif args.model_name == 'attentionUnet':
    sys.path.append('./models/unet')
    model_dir = os.path.join('./models/unet/', args.model_name)
    sys.path.append(model_dir)
    from trainer_eager import train_step, evaluate
    from model import AttentionalUnet
    segmentor_net = AttentionalUnet()

set_logger(os.path.join(model_dir, 'train_{}.log'.format(datetime.now().strftime('%m-%d_%H-%M'))))

if os.path.exists('./train_summaries'):
    shutil.rmtree('./train_summaries')
if os.path.exists('./eval_summaries'):
    shutil.rmtree('./eval_summaries')

# json_path = os.path.join(model_dir, 'params.json')
# assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

model_params = Params("./params.json")
model_params = {**model_params.__dict__, **vars(args)} # convert args to dict
# model_params.model_name = args.model_name
model_params["train_size"] = 5370 #X_train.shape[0]
model_params["eval_size"] = 250 #X_val.shape[0]
model_params["model_dir"] = model_dir

x_train_path = os.path.join(args.datasets_dir, 'imgs')
y_train_path = os.path.join(args.datasets_dir, 'gts')
x_valid_path = os.path.join(args.datasets_dir, 'val_imgs')
y_valid_path = os.path.join(args.datasets_dir, 'val_gts')

summary_writer = tf.contrib.summary.create_file_writer('./train_summaries')
summary_writer.set_as_default()

train_gen = augmented_img_and_mask_generator(x_train_path, y_train_path, batch_size=model_params["batch_size"])
valid_gen = img_and_mask_generator(x_valid_path, y_valid_path, batch_size=model_params["batch_size"])

steps_per_train_epoch = int(model_params["train_size"] / model_params["batch_size"])
steps_per_valid_epoch = int(model_params["eval_size"] / model_params["batch_size"])

# train_and_evaluate(model, model_params, summary_writer, train_gen, valid_gen, steps_per_train_epoch, steps_per_valid_epoch)
lr = model_params["learning_rate"]
current_step = 0
current_epoch = 0
max_mean_IoU = 0.0
global_step = tf.train.get_or_create_global_step()

if args.model_name == 'segan':        
    # Optimizer for segmentor
    optimizerS = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
    # Optimizer for critic
    optimizerC = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)

    epoch_critic_loss_avg = tfe.metrics.Mean()
    epoch_seg_loss_avg = tfe.metrics.Mean()
    
else:
    # Optimizer for unet
    optimizerS = tf.train.AdamOptimizer(learning_rate=lr)
    epoch_seg_loss_avg = tfe.metrics.Mean()
    
    
pbar = tqdm(total=steps_per_train_epoch)

for imgs, labels in train_gen:

    if current_step == steps_per_train_epoch:
        mean_IoU, mean_loss = evaluate(valid_gen, segmentor_net, steps_per_valid_epoch)
        current_epoch += 1
        current_step = 0
        pbar.reset()

        # if args.model_name == "segan":
        #     epoch_critic_loss_avg = tfe.metrics.Mean()
        #     epoch_seg_loss_avg = tfe.metrics.Mean()            
        # else:
        #     epoch_seg_loss_avg = tfe.metrics.Mean()
        #     valid_loss_avg = tfe.metrics.Mean()

        # if max_mean_IoU < mean_IoU:
        # max_mean_IoU = mean_IoU

        save_model_weights_dir = model_dir + '/experiments/' + 'experiment_id_' + str(model_params["experiment_id"])
        if not os.path.isdir(save_model_weights_dir):
            os.makedirs(save_model_weights_dir)
        # else: 
        #     delete_dir_content(save_model_weights_dir)

        # segmentor_net._set_inputs(img)
        print("Saving weights to ", save_model_weights_dir)
        segmentor_net.save_weights(save_model_weights_dir  + '/' + model_params["model_name"] + '_epoch_' + str(current_epoch) + '_val_meanIoU_{:.3f}_meanLoss_{:.3f}.h5'.format(mean_IoU, mean_loss))

    if current_epoch == model_params["num_epochs"] + 1:
        break

    if args.model_name == "segan":
        seg_loss, critic_loss = train_step(segmentor_net, critic_net, imgs, labels, optimizerC, optimizerS)    
        epoch_critic_loss_avg(critic_loss)

    else:
        seg_loss = train_step(segmentor_net, imgs, labels, global_step, optimizerS)    

    epoch_seg_loss_avg(seg_loss)

    tf.assign_add(global_step, 1)
    current_step += 1
    pbar.update(1)


    with tf.contrib.summary.record_summaries_every_n_global_steps(model_params["save_summary_steps"]):
            
        tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
        tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
        tf.contrib.summary.scalar("seg_loss", epoch_seg_loss_avg.result())

        if args.model_name == 'segan':
            tf.contrib.summary.scalar("critic_loss", epoch_critic_loss_avg.result())
            tf.contrib.summary.scalar("total_loss", epoch_critic_loss_avg.result() + epoch_seg_loss_avg.result())
            tf.contrib.summary.image("seg_result", tf.round(tf.sigmoid(segmentor_net(imgs))) * 255)
        else:

            seg_results = segmentor_net(tf.image.convert_image_dtype(imgs, tf.float32))
            seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
            seg_results = tf.expand_dims(seg_results, -1)
            tf.contrib.summary.image("seg_result", tf.cast(seg_results * 255, tf.uint8))