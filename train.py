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
from tqdm import tqdm_notebook as tqdm
from datetime import datetime

opts = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
config = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config)
tfe = tf.contrib.eager
print("tf version: ",  tf.__version__)

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
parser.add_argument("-s", "--save_summary_steps", type=int, default=200)
parser.add_argument("-id", "--experiment_id", type=int, required=True)
parser.add_argument("-l2", "--l2_regularizer", type=float, default=0.0)
parser.add_argument("-b1", "--beta_1", type=float, default=0.9)

args = parser.parse_args()
assert(args.model_name in ['unet', 'attentionUnet'])
assert(os.path.exists(args.datasets_dir))

sys.path.append('./models/unet')
from trainer_eager import train_step, evaluate

if args.model_name == 'unet':
    model_dir = './models/unet'
    # sys.path.append(model_dir)
    
    from base_model import Unet
    segmentor_net = Unet(l2_value=args.l2_regularizer)

elif args.model_name == 'attentionUnet':
    # sys.path.append('./models/unet')
    # model_dir = os.path.join('./models/unet/', args.model_name)
    model_dir = './models/attentionUnet'
    sys.path.append(model_dir)
    # from trainer_eager import train_step, evaluate
    from model import AttentionalUnet
    segmentor_net = AttentionalUnet(l2_value=args.l2_regularizer)

set_logger(os.path.join(model_dir, 'train_{}.log'.format(datetime.now().strftime('%m-%d_%H-%M'))))

if os.path.exists('./train_summaries'):
    shutil.rmtree('./train_summaries')
if os.path.exists('./eval_summaries'):
    shutil.rmtree('./eval_summaries')


model_params = vars(args) # convert args to dict merge with dict from params.json
model_params["model_dir"] = model_dir

x_train_path = os.path.join(args.datasets_dir, 'imgs')
y_train_path = os.path.join(args.datasets_dir, 'gts')
x_valid_path = os.path.join(args.datasets_dir, 'val_imgs')
y_valid_path = os.path.join(args.datasets_dir, 'val_gts')

num_train_imgs = len([name for name in os.listdir(os.path.join(x_train_path, 'data')) if os.path.isfile(os.path.join(x_train_path, 'data', name))])
num_train_lbls = len([name for name in os.listdir(os.path.join(y_train_path, 'data')) if os.path.isfile(os.path.join(y_train_path, 'data', name))])

assert(num_train_imgs == num_train_lbls)

num_valid_imgs = len([name for name in os.listdir(os.path.join(x_valid_path, 'data')) if os.path.isfile(os.path.join(x_valid_path, 'data', name))])
num_valid_lbls = len([name for name in os.listdir(os.path.join(y_valid_path, 'data')) if os.path.isfile(os.path.join(y_valid_path, 'data', name))])

assert(num_valid_imgs == num_valid_lbls)

model_params["train_size"] = num_train_imgs 
model_params["eval_size"] = num_valid_imgs 

summary_writer = tf.contrib.summary.create_file_writer('./train_summaries')
summary_writer.set_as_default()

train_gen = augmented_img_and_mask_generator(x_train_path, y_train_path, batch_size=model_params["batch_size"])
valid_gen = img_and_mask_generator(x_valid_path, y_valid_path, batch_size=model_params["batch_size"])

steps_per_train_epoch = int(model_params["train_size"] / model_params["batch_size"])
steps_per_valid_epoch = int(model_params["eval_size"] / model_params["batch_size"])

print("steps per train epoch", steps_per_train_epoch)
print("steps per valid epoch", steps_per_valid_epoch)
# train_and_evaluate(model, model_params, summary_writer, train_gen, valid_gen, steps_per_train_epoch, steps_per_valid_epoch)
lr = model_params["learning_rate"]
l2 = model_params["l2_regularizer"]
print("learning rate: {}".format(str(lr)))
print("l2 regularize: {}".format(str(l2)))
current_step = 0
current_epoch = 0
max_mean_IoU = 0.0
global_step = tf.train.get_or_create_global_step()

learning_rate = tfe.Variable(lr)
optimizerS = tf.train.AdamOptimizer(learning_rate=learning_rate)
epoch_seg_loss_avg = tfe.metrics.Mean()
epoch_IoU_avg = tfe.metrics.Mean()
epoch_Hd_avg = tfe.metrics.Mean()
epoch_dice_avg = tfe.metrics.Mean()
epoch_combi_avg = tfe.metrics.Mean()
    
# pbar = tqdm(total=steps_per_train_epoch)

for imgs, labels in train_gen:

    if current_step == steps_per_train_epoch:
        val_mean_IoU, val_mean_hd, val_mean_loss, val_mean_dice = evaluate(valid_gen, segmentor_net, steps_per_valid_epoch)
        current_epoch += 1
        current_step = 0
        # pbar.n = 1
        # pbar.last_print_n = 1
        epoch_seg_loss_avg = tfe.metrics.Mean()
        epoch_IoU_avg = tfe.metrics.Mean()
        epoch_Hd_avg = tfe.metrics.Mean()       
        epoch_dice_avg = tfe.metrics.Mean()

        save_model_weights_dir = model_dir + '/experiments/' + 'experiment_id_' + str(model_params["experiment_id"])
        if not os.path.isdir(save_model_weights_dir):
            os.makedirs(save_model_weights_dir)
        print("current lr ", learning_rate.numpy())
        print("Saving weights to ", save_model_weights_dir)
        segmentor_net.save_weights(save_model_weights_dir  + '/' + model_params["model_name"] + '_epoch_' + str(current_epoch) + '_val_meanIoU_{:.3f}_meanLoss_{:.3f}_meanHd_{:.3f}_meanDice_{:.3f}.h5'.format(val_mean_IoU, val_mean_loss, val_mean_hd, val_mean_dice))

    if current_epoch == model_params["num_epochs"] + 1:
        break
    
    seg_loss, batch_hd, batch_IoU, batch_dice, batch_combi = train_step(segmentor_net, imgs, labels, global_step, optimizerS)    

    if args.learning_rate_decay > 0.0:
        learning_rate.assign(tf.train.exponential_decay(lr, global_step, 5000, decay_rate=args.learning_rate_decay)())

    epoch_seg_loss_avg(seg_loss)
    epoch_Hd_avg(batch_hd)
    epoch_IoU_avg(batch_IoU)
    epoch_dice_avg(batch_dice)
    epoch_combi_avg(batch_combi)
    # tf.assign_add(global_step, 1)
    current_step += 1
    # pbar.update(1)    

    with tf.contrib.summary.record_summaries_every_n_global_steps(model_params["save_summary_steps"]):
        
        
            
        tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
        tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
        tf.contrib.summary.scalar("seg_loss", epoch_seg_loss_avg.result())
        tf.contrib.summary.scalar("Hd", epoch_Hd_avg.result())
        tf.contrib.summary.scalar("IoU", epoch_IoU_avg.result())
        tf.contrib.summary.scalar("Dice", epoch_IoU_avg.result())
        tf.contrib.summary.scalar("Combi", epoch_combi_avg.result())

        tf.contrib.summary.scalar("lr", learning_rate)

        if global_step.numpy() % model_params["save_summary_steps"] == 0:
            print("step {}, seg_loss {:.4f}, batch_hd {:.4f}, batch_IoU {:.4f}, batch_dice {:.4f}, batch_combi {:.4f}".format(current_step, seg_loss, batch_hd, batch_IoU, batch_dice, batch_combi))
    
        # seg_results = segmentor_net(tf.image.convert_image_dtype(imgs, tf.float32))
        # seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
        # seg_results = tf.expand_dims(seg_results, -1)

        # tf.contrib.summary.image("seg_result", tf.cast(seg_results * 255, tf.uint8))
