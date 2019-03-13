import tensorflow as tf
from data_loader import DataLoader


def _parse_function(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def input_fn(is_training, X, Y, params):
    # features_placeholder = tf.placeholder(X.dtype, X.shape)
    # labels_placeholder = tf.placeholder(Y.dtype, Y.shape)
    num_samples = X.shape[0]

    def parse_fn(f, l): return _parse_function(f, l)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((X, Y))
                   .shuffle(num_samples)
                   .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                   .batch(params.batch_size)
                   .prefetch(1)
                   )
    else:
        dataset = (X, Y)

    return {"dataset": dataset}
