import tensorflow as tf
import numpy as np
import os
from utils import batch_img_generator

print("tf version: ",  tf.__version__)

opts = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
config = tf.ConfigProto(gpu_options=opts)


tf.enable_eager_execution(config)

tfe = tf.contrib.eager


def train_and_evaluate(train_model_specs, val_model_specs, model_dir, params):

    # train_dataset = train_model_specs["dataset"]
    # validation_dataset = val_model_specs["dataset"]
    x_train = train_model_specs["x_train"]
    y_train = train_model_specs["y_train"]
    x_valid = val_model_specs["x_valid"]
    y_valid = val_model_specs["y_valid"]


    u_net = train_model_specs["unet"]
    
    summary_writer = tf.contrib.summary.create_file_writer('./train_summaries')
    summary_writer.set_as_default()   

    global_step = tf.train.get_or_create_global_step()

    lr = params.learning_rate

    # Optimizer for segmentor
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    maxIoU = 0
    mIoU = 0
    
    train_gen = batch_img_generator(x_train, y_train, params.num_epochs, params.batch_size)
    valid_gen = batch_img_generator(x_valid, y_valid, batch_size=params.batch_size)

    
    current_epoch = 1
    epoch_loss_avg = tfe.metrics.Mean()

    for imgs, labels, epoch in train_gen:        

        """
        At the end of every epoch, validate on validation dataset.
        And compute mean IoU.
        """
        if current_epoch < epoch:
            current_epoch = epoch
            epoch_loss_avg = tfe.metrics.Mean()
            
            IoUs = []
            valid_loss_avg = tfe.metrics.Mean()
            
            print("Validation starts.")

            for imgs, labels, _ in valid_gen:        
                imgs = tf.image.convert_image_dtype(imgs, tf.float32)
                #labels = tf.cast(labels, tf.int32)

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

            IoUs = np.array(IoUs, dtype=np.float64)
            mIoU = np.mean(IoUs, axis=0)

            # with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps):
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("val_avg_loss", valid_loss_avg.result())
                tf.contrib.summary.scalar("val_avg_IoU", mIoU)

            print("Epoch {0}, loss epoch avg {1:.4f}, loss valid avg {2:.4f}, mIoU on validation set: {3:.4f}".format(epoch, epoch_loss_avg.result(), valid_loss_avg.result(), mIoU))
                        

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

        # imgs, labels = preprocess(imgs, labels)

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

        # Summaries for tensorboard
        with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps):
            # if i % params.save_summary_steps == 0:
                        
            seg_results = tf.argmax(seg_results, axis=-1, output_type=tf.int32)
            seg_results = tf.expand_dims(seg_results, -1)

            tf.contrib.summary.image("train_img", tf.cast(imgs * 255, tf.uint8))
            tf.contrib.summary.image("ground_tr", tf.cast(labels * 255, tf.uint8))
            tf.contrib.summary.image("seg_result", tf.cast(seg_results * 255, tf.uint8))

            tf.contrib.summary.scalar("train_avg_loss", epoch_loss_avg.result())