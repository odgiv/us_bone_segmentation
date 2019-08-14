import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm_notebook as tqdm
from utils import augmented_img_and_mask_generator, img_and_mask_generator, hausdorf_distance, iou, dice, focal_loss
from datetime import datetime
import logging

    
def evaluate(valid_gen, u_net, steps_per_valid_epoch):
    IoUs = tf.contrib.eager.metrics.Mean()
    hds = tf.contrib.eager.metrics.Mean()
    dices = tf.contrib.eager.metrics.Mean()    

    valid_loss_avg = tf.contrib.eager.metrics.Mean()
    logging.info("Starting validation...")
    current_val_step = 0
    # pbar = tqdm(total = steps_per_valid_epoch)
    for imgs, labels in valid_gen:        
     
        labels[labels>=0.5] = 1.
        labels[labels<0.5] = 0.
        labels = labels.astype('uint8')
        
        imgs = tf.image.convert_image_dtype(imgs, tf.float32)

        pred = u_net(imgs)
        pred_np = pred.numpy()     
        
        pred_np = np.argmax(pred_np, axis=-1)
        pred_np = np.expand_dims(pred_np, -1)
    
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=pred)
        # loss = focal_loss(pred, labels)
        valid_loss_avg(loss)
        

        for x in range(imgs.shape[0]):
            pred_locations = np.argwhere(pred_np[x] == 1)
            label_locations = np.argwhere(labels[x] == 1)

            hd = hausdorf_distance(pred_locations, label_locations)            
            # IoU = np.sum(pred_np[x][labels[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(labels[x]) - np.sum(pred_np[x][labels[x] == 1]))
            IoU = iou(pred_np[x], labels[x])
            # dice = np.sum(pred_np[x][labels[x] == 1])*2 / float(np.sum(pred_np[x]) + np.sum(labels[x]))
            Dice = dice(pred_np[x], labels[x])            

            hds(hd)
            dices(Dice)
            IoUs(IoU)                            

        current_val_step += 1
        # pbar.update()
        if current_val_step == steps_per_valid_epoch:
            break
            

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("val_avg_hd", hds.result())
        tf.contrib.summary.scalar("val_avg_loss", valid_loss_avg.result())
        tf.contrib.summary.scalar("val_avg_IoU", IoUs.result())   
        tf.contrib.summary.scalar("val_avg_dice", dices.result())        

        tf.contrib.summary.image("valid_img", tf.cast(imgs * 255, tf.uint8))
        tf.contrib.summary.image("valid_ground_tr", tf.cast(labels * 255, tf.uint8))

        seg_results = u_net(tf.image.convert_image_dtype(imgs, tf.float32))
        seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
        seg_results = tf.expand_dims(seg_results, -1)
        tf.contrib.summary.image("val_seg_result", tf.cast(seg_results * 255, tf.uint8))
        
    combi = (100* (1 - IoUs.result()) + hds.result())
    logging.info("loss valid avg {0:.4f}, mIoU on validation set: {1:.4f}, mHd on validation set: {2:.4f}, mdice on validation set: {3:.4f}, mIoU + mHd on validation set: {4:.4f}".format(valid_loss_avg.result(), IoUs.result(), hds.result(), dices.result(), combi))
    
    return IoUs.result(), hds.result(), valid_loss_avg.result(), dices.result(), combi     


def train_step(net, imgs, labels, global_step, optimizer):
    labels[labels>=0.5] = 1.
    labels[labels<0.5] = 0.
    labels = labels.astype('uint8')

    imgs = tf.image.convert_image_dtype(imgs, tf.float32)

    with tf.GradientTape() as tape:
        # Run image through segmentor net and get result
        seg_results = net(imgs)        
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=seg_results)
        # loss = focal_loss(seg_results, labels)

    grads = tape.gradient(loss, net.trainable_variables)

    optimizer.apply_gradients(zip(grads, net.trainable_variables), global_step=global_step)
    # epoch_loss_avg(loss)

    batch_hds = []
    batch_IoUs = []
    batch_dices = []

    pred_np = seg_results.numpy() 
    batch_size = pred_np.shape[0]
    pred_np = np.argmax(pred_np, axis=-1)
    pred_np = np.expand_dims(pred_np, -1)

    for i in range(batch_size):
        label_slice = labels[i]
        pred_slice = pred_np[i]
        pred_locations = np.argwhere(pred_slice == 1)
        label_locations = np.argwhere(label_slice == 1)

        hd = hausdorf_distance(pred_locations, label_locations)
        batch_hds.append(hd)

        IoU = iou(pred_slice, label_slice)#np.sum(pred_slice[label_slice == 1]) / float(np.sum(pred_slice) + np.sum(label_slice) - np.sum(pred_slice[label_slice == 1]))
        Dice = dice(pred_slice, label_slice) #np.sum(pred_slice[label_slice == 1])*2 / float(np.sum(pred_slice) + np.sum(label_slice))

        batch_IoUs.append(IoU)
        batch_dices.append(Dice)
        
    
    combi = (100* (1 - np.mean(batch_IoUs)) + np.mean(batch_hds))

    return loss, np.mean(batch_hds), np.mean(batch_IoUs), np.mean(batch_dices), combi
