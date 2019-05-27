import tensorflow as tf
import numpy as np
import os
from utils import augmented_img_and_mask_generator, img_and_mask_generator

print("tf version: ",  tf.__version__)

opts = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
config = tf.ConfigProto(gpu_options=opts)

tf.enable_eager_execution(config)
tfe = tf.contrib.eager


def train_and_evaluate(model, x_train, y_train, x_val, y_val, params):

    u_net = model
    
    summary_writer = tf.contrib.summary.create_file_writer('./train_summaries')
    summary_writer.set_as_default()   

    global_step = tf.train.get_or_create_global_step()

    lr = params.learning_rate

    # Optimizer for segmentor
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    maxIoU = 0
    mIoU = 0
    
    # train_gen = batch_img_generator(x_train, y_train, params.num_epochs, params.batch_size)
    valid_gen = img_and_mask_generator(x_val, y_val, batch_size=params.batch_size)

    train_gen = augmented_img_and_mask_generator(x_train, y_train, params.batch_size)
    #valid_gen = batch_img_and_mask_generator(x_valid, y_valid, params.batch_size)

    steps_per_train_epoch = int(params.train_size / params.batch_size)
    steps_per_valid_epoch = int(params.eval_size / params.batch_size)

    current_epoch = 1    
    current_step = 0
    epoch_loss_avg = tfe.metrics.Mean()

    for imgs, labels in train_gen:        

        """
        At the end of every epoch, validate on validation dataset.
        And compute mean IoU.
        """
        if current_step == steps_per_train_epoch:            
            
            IoUs = []
            valid_loss_avg = tfe.metrics.Mean()
            
            print("Validation starts.")
            current_val_step = 0
            for imgs, labels in valid_gen:        
                
                # imgs = imgs.astype('float32')
                labels = (label / 255.)
                labels[labels>0] = 1
                labels[labels==0] = 0
                # labels = labels.astype('uint8')

                imgs = tf.image.convert_image_dtype(imgs, tf.float32)
                labels = tf.cast(labels, tf.int32)

                pred = u_net(imgs)
                
                gt = labels
                pred_np = pred.numpy()
                
                pred_np = np.argmax(pred_np, axis=-1)
                pred_np = np.expand_dims(pred_np, -1)

                loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=pred)
                valid_loss_avg(loss)

                for x in range(imgs.shape[0]):
                    IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))
                    IoUs.append(IoU)
                
                current_val_step += 1
                if current_val_step == steps_per_valid_epoch:
                    break

            IoUs = np.array(IoUs, dtype=np.float64)
            mIoU = np.mean(IoUs, axis=0)

            # with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps):
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("val_avg_loss", valid_loss_avg.result())
                tf.contrib.summary.scalar("val_avg_IoU", mIoU)

            print("Epoch {0}, loss epoch avg {1:.4f}, loss valid avg {2:.4f}, mIoU on validation set: {3:.4f}".format(current_epoch, epoch_loss_avg.result(), valid_loss_avg.result(), mIoU))
            current_epoch += 1
            current_step = 0
            epoch_loss_avg = tfe.metrics.Mean()
            
            if maxIoU < mIoU:
                maxIoU = mIoU

                # segmentor_net._set_inputs(img)
                print("Saving weights to ", params.save_weights_path)
                u_net.save_weights(params.save_weights_path + params.model_name + '_val_maxIoU_{:.3f}.h5'.format(maxIoU))            
                # tf.keras.models.save_model(segmentor_net, params.save_weights_path + 'segan_model_maxIoU_{:4f}.h5'.format(maxIoU), overwrite=True, include_optimizer=False)
                # tf.contrib.saved_model.save_keras_model(segmentor_net, params.save_weights_path, serving_only=True)

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
        labels = (labels / 255.)
        labels[labels>0] = 1
        labels[labels==0] = 0
        # labels = labels.astype('uint8')

        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        labels = tf.cast(labels, tf.int32)

        with tf.GradientTape() as tape:
            # Run image through segmentor net and get result
            seg_results = u_net(imgs)

            # seg_result = tf.argmax(seg_result, axis=-1, output_type=tf.float32)
            # seg_result = tf.expand_dims(seg_result, -1)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=seg_results)


        grads = tape.gradient(loss, u_net.trainable_variables)

        optimizer.apply_gradients(zip(grads, u_net.trainable_variables), global_step=global_step)
        
        epoch_loss_avg(loss)

        tf.assign_add(global_step, 1)
        current_step += 1

        # Summaries to tensorboard
        with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps):
            # if i % params.save_summary_steps == 0:
                        
            seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
            seg_results = tf.expand_dims(seg_results, -1)

            tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
            tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
            tf.contrib.summary.image("seg_result", tf.cast(seg_results * 255, tf.uint8))
            tf.contrib.summary.scalar("train_avg_loss", epoch_loss_avg.result())