import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from utils import augmented_img_and_mask_generator, img_and_mask_generator, dice_loss
from datetime import datetime
import logging

    
def evaluate(valid_gen, u_net, steps_per_valid_epoch):
    IoUs = []
    valid_loss_avg = tf.contrib.eager.metrics.Mean()
    logging.info("Starting validation...")
    current_val_step = 0
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
        valid_loss_avg(loss)

        for x in range(imgs.shape[0]):
            IoU = np.sum(pred_np[x][labels[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(labels[x]) - np.sum(pred_np[x][labels[x] == 1]))
            IoUs.append(IoU)
        
        current_val_step += 1
        if current_val_step == steps_per_valid_epoch:
            break

    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("val_avg_loss", valid_loss_avg.result())
        tf.contrib.summary.scalar("val_avg_IoU", mIoU)    
    
    print("loss valid avg {0:.4f}, mIoU on validation set: {1:.4f}".format(valid_loss_avg.result(), mIoU))
    
    return mIoU    
    

def train_and_evaluate(net, params, train_gen, valid_gen, steps_per_train_epoch, steps_per_valid_epoch):
    
    # summary_writer = tf.contrib.summary.create_file_writer('./train_summaries')
    # summary_writer.set_as_default()   

    global_step = tf.train.get_or_create_global_step()

    lr = params.learning_rate

    # Optimizer for segmentor
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    maxIoU = 0
    mIoU = 0
    
    # train_gen = batch_img_generator(x_train, y_train, params.num_epochs, params.batch_size)
    # valid_gen = batch_img_and_mask_generatox_valid, y_valid, params.batch_size)
    
    current_epoch = 1    
    current_step = 0
    epoch_loss_avg = tfe.metrics.Mean()
    
    pbar = tqdm(total=steps_per_train_epoch) 
    for imgs, labels in train_gen:        

        """
        At the end of every epoch, validate on validation dataset.
        And compute mean IoU.
        """
        if current_step == steps_per_train_epoch:            
            
            print("Epoch {0}, loss epoch avg {1:.4f}".format(current_epoch, epoch_loss_avg.result()))
            mIoU = evaluate(valid_gen, net, steps_per_valid_epoch)            
            current_epoch += 1
            current_step = 0
            pbar.reset()
            epoch_loss_avg = tfe.metrics.Mean()

            if maxIoU < mIoU:
                maxIoU = mIoU

                save_model_weights_dir = model_dir + '/model_weights/valid_img_vol_' + dataset_params["test_datasets_folder"] + '_' + datetime.now().strftime('%m-%d_%H-%M-%S') + '/'
                if not os.path.isdir(save_model_weights_dir):
                    os.makedirs(save_model_weights_dir)
                else: 
                    delete_dir_content(save_model_weights_dir)

                # segmentor_net._set_inputs(img)
                print("Saving weights to ", save_weights_path)
                net.save_weights(save_weights_path + params.model_name + '_val_maxIoU_{:.3f}.h5'.format(maxIoU))            
            
            # Learning rate decay
            # if epoch % 25 == 0:
            #     lr = lr * params.lr_decay
            #     if lr <= 0.00000001:
            #         lr = 0.00000001
            #     print("Learning rate: {:.6f}", format(lr))

            #     optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        if current_epoch == params.num_epochs + 1:
            break

        # imgs = imgs.astype('float32')
        #labels = labels / 255.
        labels[labels>=0.5] = 1.
        labels[labels<0.5] = 0.
        labels = labels.astype('uint8')

        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        #imgs = tf.convert_to_tensor(imgs, tf.float32)
        # labels = tf.cast(labels, tf.int32)

        with tf.GradientTape() as tape:
            # Run image through segmentor net and get result
            seg_results = net(imgs)        

            loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=seg_results)


        grads = tape.gradient(loss, net.trainable_variables)

        optimizer.apply_gradients(zip(grads, net.trainable_variables), global_step=global_step)
        
        epoch_loss_avg(loss)

        tf.assign_add(global_step, 1)
        current_step += 1
        pbar.update(1)

        # Summaries to tensorboard
        with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps):
            # if i % params.save_summary_steps == 0:
                        
            seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
            seg_results = tf.expand_dims(seg_results, -1)

            tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
            tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
            tf.contrib.summary.image("seg_result", tf.cast(seg_results * 255, tf.uint8))
            tf.contrib.summary.scalar("train_avg_loss", epoch_loss_avg.result())


def train_step(net, imgs, labels, global_step, optimizer):
    labels[labels>=0.5] = 1.
    labels[labels<0.5] = 0.
    labels = labels.astype('uint8')

    imgs = tf.image.convert_image_dtype(imgs, tf.float32)

    with tf.GradientTape() as tape:
        # Run image through segmentor net and get result
        seg_results = net(imgs)        
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=seg_results)

        seg_results = np.argmax(seg_results, axis=-1)
        seg_results = np.expand_dims(seg_results, -1)

        loss = dice_loss(tf.cast(seg_results, tf.int32), tf.cast(labels, tf.int32))

    grads = tape.gradient(loss, net.trainable_variables)

    optimizer.apply_gradients(zip(grads, net.trainable_variables), global_step=global_step)
    # epoch_loss_avg(loss)
    return loss