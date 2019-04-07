import tensorflow as tf
import numpy as np
import os
print("tf version: ",  tf.__version__)

config = tf.ConfigProto(allow_soft_placement=True)
config.per_process_gpu_memory_fraction = 0.4

tf.enable_eager_execution(config)

tfe = tf.contrib.eager


def train_and_evaluate(train_model_specs, val_model_specs, model_dir, params):

    if not os.path.isdir(params.save_weights_path):
        os.mkdir(params.save_weights_path)

    train_dataset = train_model_specs["dataset"]
    validation_dataset = val_model_specs["dataset"]
    u_net = train_model_specs["unet"]
    
    summary_writer = tf.contrib.summary.create_file_writer('./train_summaries', flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    lr = params.learning_rate

    # Optimizer for segmentor
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    maxIoU = 0
    mIoU = 0

    epoch = 1
    for epoch in range(1, params.num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        
        for i, (img, label) in enumerate(train_dataset):

            # print(len(segmentor_net.trainable_variables))
            # make_trainable(segmentor_net, True)
            # print(len(segmentor_net.trainable_variables))
            # print(len(critic_net.trainable_variables))

            label = tf.cast(label, tf.int32)

            with tf.GradientTape() as tape:
                # Run image through segmentor net and get result
                seg_result = u_net(img)

                # seg_result = tf.argmax(seg_result, axis=-1, output_type=tf.float32)
                # seg_result = tf.expand_dims(seg_result, -1)

                loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=seg_result)


            grads = tape.gradient(loss, u_net.trainable_variables)

            optimizer.apply_gradients(zip(grads, u_net.trainable_variables), global_step=global_step)
            
            epoch_loss_avg(loss)

            tf.assign_add(global_step, 1)


        """
        At the end of every epoch, validate on validation dataset.
        And compute mean IoU.
        """
        IoUs = []
        valid_loss_avg = tfe.metrics.Mean()
        print("Validation starts.")
        for i, (imgs, labels) in enumerate(validation_dataset):
            pred = u_net(imgs)
            
            gt = labels.numpy()
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

        print("Epoch {0}, loss epoch avg {1:.4f}, loss valid avg {2:.4f}, mIoU on validation set: {3:.4f}".format(epoch, epoch_loss_avg.result(), valid_loss_avg.result(), mIoU))
        # print(global_step)
        # Summaries for tensorboard
        #with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps, global_step=global_step):
            # if i % params.save_summary_steps == 0:
                        
            # seg_result = tf.argmax(seg_result, axis=-1, output_type=tf.int32)
            # seg_result = tf.expand_dims(seg_result, -1)

            # tf.contrib.summary.image("train_img", img)
            # tf.contrib.summary.image("ground_tr", tf.cast(label * 255, tf.uint8))
            # tf.contrib.summary.image("seg_result", tf.cast(seg_result * 255, tf.uint8))

        tf.contrib.summary.scalar("train_avg_loss", epoch_loss_avg.result())
        tf.contrib.summary.scalar("val_avg_loss", valid_loss_avg.result())

        if maxIoU < mIoU:
            maxIoU = mIoU

            # segmentor_net._set_inputs(img)
            u_net.save_weights(params.save_weights_path + 'unet_val_maxIoU_{:.3f}.h5'.format(maxIoU))            
            # tf.keras.models.save_model(segmentor_net, params.save_weights_path + 'segan_model_maxIoU_{:4f}.h5'.format(maxIoU), overwrite=True, include_optimizer=False)
            # tf.contrib.saved_model.save_keras_model(segmentor_net, params.save_weights_path, serving_only=True)

        # Learning rate decay
        if epoch % 25 == 0:
            lr = lr * params.lr_decay
            if lr <= 0.00000001:
                lr = 0.00000001
            print("Learning rate: {:.6f}", format(lr))

            optimizer = tf.train.AdamOptimizer(learning_rate=lr)