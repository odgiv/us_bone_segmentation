import logging
import os
from tqdm import trange

import tensorflow as tf

def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):
    update_metrics = model_spec["update_metrics"]
    eval_metrics = model_spec["metrics"]

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

    for _ in range(num_steps):
        sess.run(update_metrics)
    
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)
    
    return metrics_val

def evaluate(model_spec, model_dir, params):

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(model_spec['variable_init_op'])
        sess.run(model_spec['local_variable_init_op'])

        num_steps = (params.eval_size + params.batch_size -1) // params.batch_size
        metrics = evaluate_sess(sess, model_spec, num_steps)
