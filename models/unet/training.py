import argparse
import logging
from tqdm import trange

import tensorflow as tf

def train_and_evaluate(train_model_spec, eval_model_spec, params):
    """Train the model and evaluate every epoch
    """
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1) # only keep 1 best checkpoint (best on eval)
    num_steps = (params.train_size + params.batch_size -1) // params.batch_size
    with tf.Session() as sess:
        sess.run(train_model_spec['variable_init_op'])

        best_eval_acc = 0.0
        
        for epoch in range(params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            loss = train_model_spec["loss"]
            train_op = train_model_spec["train_op"]
            # update_metrics = model_spec["update_metricts"]
            # metrics = model_spec["metrics"]
            # summary_op = model_spec["summapy_op"]
            global_step = tf.train.get_global_step()
            iterator_init_op = train_model_spec["iterator_init_op"]
            # metric_init_op = model_spec["metrics_init_op"]            
            features_placeholder = train_model_spec["X_placeholder"]
            labels_placeholder = train_model_spec["Y_placeholder"]

            sess.run(iterator_init_op, feed_dict={
                features_placeholder: train_model_spec["X"],
                labels_placeholder: train_model_spec["Y"]
            })
            #sess.run(metric_init_op)

            # Use tqdm for progress bar
            t = trange(num_steps)

            for i in t:
                imgs = train_model_spec["images"]
                lbls = train_model_spec["labels"]

                sess.run([imgs, lbls])

                #print(imgs.shape, lbls.shape)
                _, loss_val = sess.run([train_op, loss])
                t.set_postfix(loss='{:05.3f}'.format(loss_val))





            