import tensorflow as tf
def mean_squared_error( labels, logit):
    loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=logit)
    return loss