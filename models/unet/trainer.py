import argparse
import logging
import os
import shutil
from tqdm import trange

import tensorflow as tf



def train_sess(sess, model_spec, num_steps, writer, params):
    loss = model_spec["loss"]
    mean_iou = model_spec["mean_iou"]
    conf_mat = model_spec["conf_mat"]
    train_op = model_spec["train_op"]
    update_metrics = model_spec["update_metrics"]
    metrics = model_spec["metrics"]
    summary_op = model_spec["summary_op"]

    global_step = tf.train.get_global_step()
    iterator_init_op = model_spec["iterator_init_op"]
    metric_init_op = model_spec["metrics_init_op"]            

    features_placeholder = model_spec["X_placeholder"]
    labels_placeholder = model_spec["Y_placeholder"]

    sess.run(metric_init_op)
    sess.run(iterator_init_op, feed_dict={
        features_placeholder: model_spec["X"],
        labels_placeholder: model_spec["Y"]
    })

    # Use tqdm for progress bar
    t = trange(num_steps)

    for i in t:
        # imgs = train_model_spec["images"]
        # lbls = train_model_spec["labels"]

        # sess.run([imgs, lbls])
        #print(imgs.shape, lbls.shape)

        if i % params.save_summary_steps == 0:
            _, _, _, loss_val, mean_iou_val, summ, global_step_val = sess.run([train_op, update_metrics, conf_mat, loss, mean_iou, summary_op, global_step])
            writer.add_summary(summ, global_step_val)

        else:
            _, _, _, loss_val, mean_iou_val = sess.run([train_op, update_metrics, conf_mat, loss, mean_iou])
        t.set_postfix(loss='{:05.3f}'.format(loss_val), mean_iou='{:05.3f}'.format(mean_iou_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(train_model_spec, eval_model_spec, model_dir, params):
    """Train the model and evaluate every epoch
    """
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1) # only keep 1 best checkpoint (best on eval)
    num_steps = (params.train_size + params.batch_size -1) // params.batch_size

    if os.path.exists('./train_summaries'):
        shutil.rmtree('./train_summaries')
    if os.path.exists('./eval_summaries'):
        shutil.rmtree('./eval_summaries')

    with tf.Session() as sess:
        sess.run(train_model_spec['variable_init_op'])
        sess.run(train_model_spec['local_variable_init_op'])

        train_writer = tf.summary.FileWriter('./train_summaries', sess.graph)
        eval_writer = tf.summary.FileWriter('./eval_summaries', sess.graph)

        best_eval_acc = 0.0
        for epoch in range(params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            train_sess(sess, train_model_spec, num_steps, train_writer, params)





            