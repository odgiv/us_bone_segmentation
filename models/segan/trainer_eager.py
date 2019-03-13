import tensorflow as tf
import numpy as np
print("tf version: ",  tf.__version__)
tf.enable_eager_execution()

tfe = tf.contrib.eager


def make_trainable(net, val):
    """
    Set layers in keras model trainable or not

    val: trainable True or False
    """
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def train_and_evaluate(train_model_specs, val_model_specs, model_dir, params):
    train_dataset = train_model_specs["dataset"]
    segmentor_net = train_model_specs["segmentor_net"]
    critic_net = train_model_specs["critic_net"]
    validation_dataset = val_model_specs["dataset"]

    summary_writer = tf.contrib.summary.create_file_writer(
        './train_summaries', flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    # Optimizer for segmentor
    optimizerS = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # Optimizer for critic
    optimizerC = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    maxIoU = 0

    for epoch in range(params.num_epochs):
        epoch_critic_loss_avg = tfe.metrics.Mean()
        epoch_seg_loss_avg = tfe.metrics.Mean()

        for i, (img, label) in enumerate(train_dataset):

            make_trainable(critic_net, True)

            make_trainable(segmentor_net, False)

            # print(len(segmentor_net.trainable_variables))
            # make_trainable(segmentor_net, True)
            # print(len(segmentor_net.trainable_variables))
            # print(len(critic_net.trainable_variables))

            with tf.GradientTape() as tape:
                # Run image through segmentor net and get result
                seg_result = segmentor_net(img)
                seg_result = tf.sigmoid(seg_result)
                seg_result_masked = img * seg_result
                target_masked = img * label

                critic_result_on_seg = critic_net(seg_result_masked)
                critic_result_on_target = critic_net(target_masked)

                critic_loss = - \
                    tf.reduce_mean(
                        tf.abs(critic_result_on_seg - critic_result_on_target))

            grads = tape.gradient(
                critic_loss, critic_net.trainable_variables)

            optimizerC.apply_gradients(
                zip(grads, critic_net.trainable_variables), global_step=global_step)

            for critic_weight in critic_net.trainable_weights:
                tf.clip_by_value(critic_weight, -0.05, 0.05)

            epoch_critic_loss_avg(critic_loss)

            make_trainable(segmentor_net, True)
            make_trainable(critic_net, False)

            with tf.GradientTape() as tape:
                seg_result = segmentor_net(img)
                seg_result = tf.sigmoid(seg_result)
                seg_result_masked = img * seg_result
                target_masked = img * label

                critic_result_on_seg = critic_net(seg_result_masked)
                critic_result_on_target = critic_net(target_masked)

                seg_loss = tf.reduce_mean(
                    tf.abs(critic_result_on_seg - critic_result_on_target))

            grads = tape.gradient(
                seg_loss, segmentor_net.trainable_variables)

            optimizerS.apply_gradients(
                zip(grads, segmentor_net.trainable_variables), global_step=global_step)

            epoch_seg_loss_avg(seg_loss)

            tf.assign_add(global_step, 1)

            with tf.contrib.summary.record_summaries_every_n_global_steps(params.save_summary_steps, global_step=global_step):
                if i % params.save_summary_steps == 0:
                    print("Epoch {0}, Step {1}, Critic loss epoch avg {2:.4f}, Seg loss epoch avg {3:.4f}, Total loss epoch avg {4:.4f}".format(
                        epoch, i, epoch_critic_loss_avg.result(), epoch_seg_loss_avg.result(), (epoch_critic_loss_avg.result() + epoch_seg_loss_avg.result())))

                # seg_result = segmentor_net(img)
                # seg_result = tf.sigmoid(seg_result)

                tf.contrib.summary.image("train_img", img)
                tf.contrib.summary.image("ground_tr", label * 255)
                tf.contrib.summary.image(
                    "seg_result", tf.round(seg_result) * 255)

                tf.contrib.summary.scalar("critic_loss",
                                          epoch_critic_loss_avg.result())
                tf.contrib.summary.scalar("seg_loss",
                                          epoch_seg_loss_avg.result())

                tf.contrib.summary.scalar(
                    "total_loss", epoch_critic_loss_avg.result() + epoch_seg_loss_avg.result())

        """
        At the end of every epoch, validate on validation dataset.
        And compute mean IoU.
        """
        IoUs = []
        print("Validation starts.")
        for i, (imgs, labels) in enumerate(validation_dataset):
            pred = segmentor_net(imgs)
            gt = labels.numpy()
            pred_np = pred.numpy()

            pred_np[pred_np <= 0.5] = 0
            pred_np[pred_np > 0.5] = 1

            for x in range(imgs.shape[0]):
                IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]
                                                                    ) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))
                IoUs.append(IoU)

        IoUs = np.array(IoUs, dtype=np.float64)
        mIoU = np.mean(IoUs, axis=0)
        print('mIoU on validation set: {:.4f}'.format(mIoU))

        if maxIoU < mIoU:
            maxIoU = mIoU
            segmentor_net.save_weights(
                params.save_weights_path + 'segan_weights' + maxIoU + '.h5')
