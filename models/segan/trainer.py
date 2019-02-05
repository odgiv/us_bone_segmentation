import logging
import os
import tensorflow as tf
from tqdm import trange


def make_trainable(net, val):
    """
    Set layers in keras model trainable or not

    val: trainable True or False
    """
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def train_sess(sess, model_spec, num_steps, writer, params):
    """
    First, train critic one step and then train segmentor
    one step
    """
    #model = model_spec["model"]
    segmentor_net = model_spec["segmentor_net"]
    critic_net = model_spec["critic_net"]
    seg_loss = model_spec["seg_loss"]
    cri_loss = model_spec["cri_loss"]
    seg_train_op = model_spec["seg_train_op"]
    cri_train_op = model_spec["cri_train_op"]
    summary_op = model_spec["summary_op"]

    global_step = tf.train.get_global_step()
    iterator_init_op = model_spec["iterator_init_op"]

    features_placeholder = model_spec["X_placeholder"]
    labels_placeholder = model_spec["Y_placeholder"]

    sess.run(iterator_init_op, feed_dict={
        features_placeholder: model_spec["X"],
        labels_placeholder: model_spec["Y"]
    })

    t = trange(num_steps)

    for i in t:

        make_trainable(critic_net, True)  # Make critic trainable
        make_trainable(segmentor_net, False)  # Detach segmentor

        _, c_loss = sess.run([cri_train_op, cri_loss])

        trainable_weights = critic_net.trainable_weights
        for weight_var in trainable_weights:
            tf.clip_by_value(weight_var, -0.05, 0.05)

        # make_trainable(critic_net, False)  # Make critic trainable
        make_trainable(segmentor_net, True)  # Detach segmentor

        _, s_loss = sess.run([seg_train_op, seg_loss])

        t.set_postfix(seg_loss='{:05.3f}'.format(
            s_loss), cri_loss='{:05.3f}'.format(c_loss))

        if i % params.save_summary_steps == 0:
            summ, global_step_val = sess.run([summary_op, global_step])
            writer.add_summary(summ, global_step_val)


def train_and_evaluate(train_model_spec, eval_model_specs, model_dir, params):

    with tf.Session() as sess:
        sess.run(train_model_spec['variable_init_op'])
        sess.run(train_model_spec['local_variable_init_op'])

        train_writer = tf.summary.FileWriter('./train_summaries', sess.graph)

        for epoch in range(params.num_epochs):
            num_steps = (params.train_size + params.batch_size -
                         1) // params.batch_size

            train_sess(sess, train_model_spec, num_steps, train_writer, params)
