import tensorflow as tf
def huber( labels, logit):
    loss = tf.losses.huber_loss(labels, logit)
    return loss