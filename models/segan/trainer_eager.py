import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import augmented_img_and_mask_generator, img_and_mask_generator
from datetime import datetime


def make_trainable(net, val):
    """
    Set layers in keras model trainable or not

    val: trainable True or False
    """
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def evaluate(valid_gen, segmentor_net, steps_per_valid_epoch):
    IoUs = []

    print("Validation starts.")
    current_val_step = 0
    for imgs, labels in valid_gen:

        labels[labels>0.] = 1.
        labels[labels==0.] = 0.
        labels = labels.astype('uint8')

        imgs = tf.image.convert_image_dtype(imgs, tf.float32)

        pred = segmentor_net(imgs)
        gt = labels
        pred_np = pred.numpy()

        pred_np[pred_np <= 0.5] = 0
        pred_np[pred_np > 0.5] = 1

        for x in range(imgs.shape[0]):
            IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))
            IoUs.append(IoU)

        current_val_step += 1
        if current_val_step == steps_per_valid_epoch:
            break

    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("val_avg_IoU", mIoU)

    print("mIoU on validation set: {0:.4f}".format(mIoU))

    return mIoU


def train_step(segmentor_net, critic_net, imgs, labels, optimizerC, optimizerS):
        make_trainable(critic_net, True)
        make_trainable(segmentor_net, False)

        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
            
        labels[labels>0.] = 1.
        labels[labels==0.] = 0.
        labels = labels.astype('uint8')

        with tf.GradientTape() as tape:
            # Run image through segmentor net and get result
            seg_result = segmentor_net(imgs)
            
            seg_result = tf.sigmoid(seg_result)
            seg_result_masked = imgs * seg_result
            target_masked = imgs * labels

            critic_result_on_seg = critic_net(seg_result_masked)
            critic_result_on_target = critic_net(target_masked)

            critic_loss = - tf.reduce_mean(tf.abs(critic_result_on_seg - critic_result_on_target))

        grads = tape.gradient(critic_loss, critic_net.trainable_variables)

        optimizerC.apply_gradients(zip(grads, critic_net.trainable_variables), global_step=global_step)

        for critic_weight in critic_net.trainable_weights:
            tf.clip_by_value(critic_weight, -0.05, 0.05)

        # epoch_critic_loss_avg(critic_loss)

        make_trainable(segmentor_net, True)
        make_trainable(critic_net, False)

        with tf.GradientTape() as tape:
            seg_result = segmentor_net(imgs)
            seg_result_sigm = tf.sigmoid(seg_result)
            seg_result_masked = imgs * seg_result_sigm
            target_masked = imgs * labels

            critic_result_on_seg = critic_net(seg_result_masked)
            critic_result_on_target = critic_net(target_masked)

            seg_loss = tf.reduce_mean(tf.abs(critic_result_on_seg - critic_result_on_target))

        grads = tape.gradient(seg_loss, segmentor_net.trainable_variables)

        optimizerS.apply_gradients(zip(grads, segmentor_net.trainable_variables), global_step=global_step)

        # epoch_seg_loss_avg(seg_loss)
        return seg_loss, critic_loss

def train_and_evaluate(net, params):
    
    segmentor_net = net.segNet
    critic_net = net.criNet
    
    summary_writer = tf.contrib.summary.create_file_writer('./train_summaries')
    summary_writer.set_as_default()

    global_step = tf.train.get_or_create_global_step()

    lr = params.learning_rate

    maxIoU = 0
    mIoU = 0
    
    # Optimizer for segmentor
    optimizerS = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.beta1)
    # Optimizer for critic
    optimizerC = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.beta1)
    
    valid_gen = img_and_mask_generator(x_val, y_val, batch_size=params.batch_size)

    train_gen = augmented_img_and_mask_generator(x_train, y_train, params.batch_size)

    steps_per_train_epoch = int(params.train_size / params.batch_size)
    steps_per_valid_epoch = int(params.eval_size / params.batch_size)

    current_epoch = 1    
    current_step = 0
    
    epoch_critic_loss_avg = tfe.metrics.Mean()
    epoch_seg_loss_avg = tfe.metrics.Mean()

    pbar = tqdm(total=steps_per_train_epoch) 
    for imgs, labels in train_gen:

        if current_step == steps_per_train_epoch:
            print("Epoch {0}, seg loss epoch avg {1:.4f}, cri loss epoch avg {2:.4f}".format(current_epoch, epoch_seg_loss_avg.result(), epoch_critic_loss_avg.result()))
            mIoU = evaluate(valid_gen, segmentor_net, steps_per_valid_epoch)            
            current_epoch += 1
            current_step = 0
            pbar.reset()
            epoch_seg_loss_avg = tfe.metrics.Mean()
            epoch_critic_loss_avg = tfe.metrics.Mean()

            if maxIoU < mIoU:
                maxIoU = mIoU

                save_model_weights_dir = model_dir + '/model_weights/valid_img_vol_' + dataset_params["test_datasets_folder"] + '_' + datetime.now().strftime('%m-%d_%H-%M-%S') + '/'
                if not os.path.isdir(save_model_weights_dir):
                    os.makedirs(save_model_weights_dir)
                else: 
                    delete_dir_content(save_model_weights_dir)

                # segmentor_net._set_inputs(img)
                print("Saving weights to ", save_weights_path)
                segmentor_net.save_weights(save_weights_path + params.model_name + '_val_maxIoU_{:.3f}.h5'.format(maxIoU))            
            
        if current_epoch == params.num_epochs + 1:
            break

        make_trainable(critic_net, True)

        make_trainable(segmentor_net, False)

        # print(len(segmentor_net.trainable_variables))
        # make_trainable(segmentor_net, True)
        # print(len(segmentor_net.trainable_variables))
        # print(len(critic_net.trainable_variables))
        
        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        
        labels[labels>0.] = 1.
        labels[labels==0.] = 0.
        labels = labels.astype('uint8')

        with tf.GradientTape() as tape:
            # Run image through segmentor net and get result
            seg_result = segmentor_net(imgs)
            
            seg_result = tf.sigmoid(seg_result)
            seg_result_masked = imgs * seg_result
            target_masked = imgs * labels

            critic_result_on_seg = critic_net(seg_result_masked)
            critic_result_on_target = critic_net(target_masked)

            critic_loss = - tf.reduce_mean(tf.abs(critic_result_on_seg - critic_result_on_target))

        grads = tape.gradient(critic_loss, critic_net.trainable_variables)

        optimizerC.apply_gradients(zip(grads, critic_net.trainable_variables), global_step=global_step)

        for critic_weight in critic_net.trainable_weights:
            tf.clip_by_value(critic_weight, -0.05, 0.05)

        epoch_critic_loss_avg(critic_loss)

        make_trainable(segmentor_net, True)
        make_trainable(critic_net, False)

        with tf.GradientTape() as tape:
            seg_result = segmentor_net(imgs)
            seg_result_sigm = tf.sigmoid(seg_result)
            seg_result_masked = imgs * seg_result_sigm
            target_masked = imgs * labels

            critic_result_on_seg = critic_net(seg_result_masked)
            critic_result_on_target = critic_net(target_masked)

            seg_loss = tf.reduce_mean(tf.abs(critic_result_on_seg - critic_result_on_target))

        grads = tape.gradient(seg_loss, segmentor_net.trainable_variables)

        optimizerS.apply_gradients(zip(grads, segmentor_net.trainable_variables), global_step=global_step)

        epoch_seg_loss_avg(seg_loss)

        tf.assign_add(global_step, 1)
        current_step += 1
        pbar.update(1)

        with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps):
            
            tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
            tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
            tf.contrib.summary.image("seg_result", tf.round(seg_result_sigm) * 255)

            tf.contrib.summary.scalar("critic_loss", epoch_critic_loss_avg.result())
            tf.contrib.summary.scalar("seg_loss", epoch_seg_loss_avg.result())

            tf.contrib.summary.scalar("total_loss", epoch_critic_loss_avg.result() + epoch_seg_loss_avg.result())

    # Learning rate decay
    # if epoch % 25 == 0:
    #     lr = lr * params.lr_decay
    #     if lr <= 0.00000001:
    #         lr = 0.00000001
    #     print("Learning rate: {:.6f}", format(lr))

    #     optimizerS = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.beta1)
    #     optimizerC = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.beta1)
