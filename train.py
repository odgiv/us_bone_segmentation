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
python train.py -m unet -d /where/your/dataset/exists/
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
parser.add_argument("-ds", "--decay_step", type=int, default=5000)
parser.add_argument("-c", "--continue_training", dest="continue_training", default=False, action='store_true')

args = parser.parse_args()
assert(args.model_name in ['unet', 'attentionUnet'])
assert(os.path.exists(args.datasets_dir))

sys.path.append('./models/unet')
from trainer_eager import train_step, evaluate

if args.model_name == 'unet':
    model_dir = './models/unet'
    from base_model import Unet
    segmentor_net = Unet(l2_value=args.l2_regularizer)

elif args.model_name == 'attentionUnet':
    model_dir = './models/attentionUnet'
    sys.path.append(model_dir)
    from model import AttentionalUnet
    segmentor_net = AttentionalUnet(l2_value=args.l2_regularizer)

set_logger(os.path.join(model_dir, 'train_{}_exp_id_{}_date_{}.log'.format(args.model_name, str(args.experiment_id), datetime.now().strftime('%y-%m-%d_%H-%M'))))

if os.path.exists('./train_summaries'):
    shutil.rmtree('./train_summaries')
if os.path.exists('./eval_summaries'):
    shutil.rmtree('./eval_summaries')


model_params = vars(args) # convert args to dictionary
model_params["model_dir"] = model_dir

# The following directories have to exist in datasets_dir.
# In each directory, another sub directory called "data" must exist. 
# Training and validation images must be in this "data" sub directory.  
x_train_path = os.path.join(args.datasets_dir, 'train_imgs')
y_train_path = os.path.join(args.datasets_dir, 'train_gts')
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

logging.info("steps per train epoch {}".format(str(steps_per_train_epoch)))
logging.info("steps per valid epoch {}".format(str(steps_per_valid_epoch)))

lr = model_params["learning_rate"]
l2 = model_params["l2_regularizer"]
logging.info("learning rate: {}".format(str(lr)))
logging.info("l2 regularize: {}".format(str(l2)))
current_step = 0
current_epoch = 0
max_mean_IoU = 0.0
global_step = tf.train.get_or_create_global_step()

learning_rate = tfe.Variable(lr)
optimizerS = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=model_params["beta_1"])
epoch_seg_loss_avg = tfe.metrics.Mean()
epoch_IoU_avg = tfe.metrics.Mean()
epoch_Hd_avg = tfe.metrics.Mean()
epoch_dice_avg = tfe.metrics.Mean()

    
for imgs, labels in train_gen:

    if current_step == steps_per_train_epoch:
        logging.info("current epoch: {}".format(current_epoch))
        val_mean_IoU, val_mean_hd, val_mean_loss, val_mean_dice, val_combi = evaluate(valid_gen, segmentor_net, steps_per_valid_epoch)
        current_epoch += 1
        current_step = 0        
        epoch_seg_loss_avg = tfe.metrics.Mean()
        epoch_IoU_avg = tfe.metrics.Mean()
        epoch_Hd_avg = tfe.metrics.Mean()       
        epoch_dice_avg = tfe.metrics.Mean()

        save_model_weights_dir = model_dir + '/experiments/' + 'experiment_id_' + str(model_params["experiment_id"])
        if not os.path.isdir(save_model_weights_dir):
            os.makedirs(save_model_weights_dir)

        logging.info("current lr {}".format(str(learning_rate.numpy())))
        logging.info("Saving weights to {} ".format(save_model_weights_dir))
        segmentor_net.save_weights(save_model_weights_dir  + '/' + model_params["model_name"] + '_epoch_' + str(current_epoch) + '_val_meanIoU_{:.3f}_meanLoss_{:.3f}_meanHd_{:.3f}_meanDice_{:.3f}_mCombi_{:.3f}.h5'.format(val_mean_IoU, val_mean_loss, val_mean_hd, val_mean_dice, val_combi))

    if current_epoch == model_params["num_epochs"] + 1:
        break
    
    seg_loss, batch_hd, batch_IoU, batch_dice = train_step(segmentor_net, imgs, labels, global_step, optimizerS)    

    if args.learning_rate_decay > 0.0:
        learning_rate.assign(tf.train.exponential_decay(lr, global_step, model_params["decay_step"], decay_rate=args.learning_rate_decay)())

    epoch_seg_loss_avg(seg_loss)
    epoch_Hd_avg(batch_hd)
    epoch_IoU_avg(batch_IoU)
    epoch_dice_avg(batch_dice)    
    current_step += 1    

    with tf.contrib.summary.record_summaries_every_n_global_steps(model_params["save_summary_steps"]):       
            
        tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
        tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
        tf.contrib.summary.scalar("seg_loss", epoch_seg_loss_avg.result())
        tf.contrib.summary.scalar("Hd", epoch_Hd_avg.result())
        tf.contrib.summary.scalar("IoU", epoch_IoU_avg.result())
        tf.contrib.summary.scalar("Dice", epoch_IoU_avg.result())        

        tf.contrib.summary.scalar("lr", learning_rate)

        if global_step.numpy() % model_params["save_summary_steps"] == 0:
            logging.info("step {}, seg_loss {:.4f}, batch_hd {:.4f}, batch_IoU {:.4f}, batch_dice {:.4f}, batch_combi {:.4f}".format(current_step, seg_loss, batch_hd, batch_IoU, batch_dice, batch_combi))
    
        seg_results = segmentor_net(tf.image.convert_image_dtype(imgs, tf.float32))
        seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
        seg_results = tf.expand_dims(seg_results, -1)

        tf.contrib.summary.image("seg_result", tf.cast(seg_results * 255, tf.uint8))