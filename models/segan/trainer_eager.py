import tensorflow as tf
print(tf.__version__)
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


def train_and_evaluate(train_model_specs, eval_model_specs, model_dir, params):
    dataset = train_model_specs["dataset"]
    segmentor_net = train_model_specs["segmentor_net"]
    critic_net = train_model_specs["critic_net"]
    summary_writer = tf.contrib.summary.create_file_writer(
        './train_summaries', flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    optimizerS = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    optimizerC = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    for epoch in range(params.num_epochs):
        epoch_critic_loss_avg = tfe.metrics.Mean()
        epoch_seg_loss_avg = tfe.metrics.Mean()

        for i, (img, label) in enumerate(dataset):

            make_trainable(critic_net, True)

            make_trainable(segmentor_net, False)

            # print(len(segmentor_net.trainable_variables))
            # make_trainable(segmentor_net, True)
            # print(len(segmentor_net.trainable_variables))
            # print(len(critic_net.trainable_variables))

            with tf.GradientTape() as tape:
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
                    print("Epoch {}, Step {}, Critic loss epoch avg {}, Seg loss epoch avg {}".format(
                        epoch, i, epoch_critic_loss_avg.result(), epoch_seg_loss_avg.result()))

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
