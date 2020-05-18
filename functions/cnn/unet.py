##################################################
## {Description}
##################################################
## {License_info}
##################################################
## Author: {Sahar Yousefi}
## Copyright: Copyright {2020}, {LUMC}
## Credits: [Sahar Yousefi]
## License: {GPL}
## Version: 1.0.0
## Mmaintainer: {Sahar Yousefi}
## Email: {s.yousefi.radi[at]lumc.nl}
## Status: {Research}
##################################################
import tensorflow as tf
class unet:
    def __init__(self, ):
        print(1)
    def conv_batch_activation(self,variable_scope,input,filters,kernel_size=3,padding='valid',is_training_bn=True):
        with tf.variable_scope(variable_scope):
            conv = tf.layers.conv2d(input,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        activation=None,
                                        name=variable_scope+'_conv')
            bn = tf.layers.batch_normalization(conv, training=is_training_bn, renorm=False, name=variable_scope+'_bn')
            bn = tf.nn.leaky_relu(bn)
            return bn

    def unet(self,t1, asl, pet, is_training_bn):
        augmented_data=[]
        augmented_data.append(asl)
        augmented_data.append(t1)
        augmented_data.append(pet)
        t1_cba1 = self.conv_batch_activation( variable_scope='t1_early_feature_extractor_1', input=t1, filters=8,
                                              kernel_size=3, padding='valid', is_training_bn=is_training_bn)
        t1_cba2 = self.conv_batch_activation(variable_scope='t1_early_feature_extractor_2', input=t1_cba1, filters=16,
                                             kernel_size=3,padding='valid', is_training_bn=is_training_bn)

        asl_cba1 = self.conv_batch_activation(variable_scope='asl_early_feature_extractor_1', input=asl, filters=8, kernel_size=3,
                                          padding='valid', is_training_bn=is_training_bn)
        asl_cba2 = self.conv_batch_activation(variable_scope='asl_early_feature_extractor_2', input=asl_cba1, filters=16,
                                          kernel_size=3,padding='valid', is_training_bn=is_training_bn)


        with tf.variable_scope('fusion'):
            fused_feature = tf.concat([t1_cba2, asl_cba2], -1)
        t1asl_cba = self.conv_batch_activation(variable_scope='t1_asl_fusion', input=fused_feature, filters=16,
                                              kernel_size=3,padding='valid', is_training_bn=is_training_bn)

        with tf.variable_scope('pool1'):
            pool1 = tf.layers.max_pooling2d(inputs=t1asl_cba, pool_size=2, strides=2)

        fused_cba1 = self.conv_batch_activation(variable_scope='feature_extractor_1_level_2', input=pool1, filters=16,
                                             kernel_size=3, padding='same', is_training_bn=is_training_bn)
        fused_cba2 = self.conv_batch_activation(variable_scope='feature_extractor_2_level_2', input=fused_cba1, filters=32,
                                             kernel_size=3, padding='same', is_training_bn=is_training_bn)
        with tf.variable_scope('pool2'):
            pool2 = tf.layers.max_pooling2d(inputs=fused_cba2, pool_size=2, strides=2)

        fused_cba3 = self.conv_batch_activation(variable_scope='feature_extractor_1_level_3', input=pool2, filters=64,
                                                kernel_size=3, padding='same', is_training_bn=is_training_bn)
        fused_cba4 = self.conv_batch_activation(variable_scope='feature_extractor_2_level_3', input=fused_cba3,
                                                filters=128, kernel_size=3, padding='same', is_training_bn=is_training_bn)
        #upsampling
        deconv1 = tf.layers.conv2d_transpose(fused_cba4,
                                             filters=64,
                                             kernel_size=3,
                                             strides=(2,  2),
                                             padding='valid',
                                             use_bias=False)
        # des_size=23#tf.shape(fused_cba2)[1]
        # src_size=35#tf.shape(deconv1)[1]
        #
        # src=fused_cba2[:,
        # tf.to_int32(src_size / 2) - tf.to_int32(des_size / 2) - 1:
        # tf.to_int32(src_size / 2) + tf.to_int32(des_size / 2),
        # tf.to_int32(src_size / 2) - tf.to_int32(des_size / 2) - 1:
        # tf.to_int32(src_size / 2) + tf.to_int32(des_size / 2),:]

        conc1 = tf.concat([fused_cba2, deconv1], -1)
        #conv
        fused_cba5 = self.conv_batch_activation(variable_scope='feature_extractor_11_level_2', input=conc1, filters=64,
                                                kernel_size=3, padding='same', is_training_bn=is_training_bn)
        fused_cba6 = self.conv_batch_activation(variable_scope='feature_extractor_22_level_2', input=fused_cba5,
                                                filters=16, kernel_size=3, padding='same',
                                                is_training_bn=is_training_bn)
        # upsampling
        deconv2 = tf.layers.conv2d_transpose(fused_cba6,
                                             filters=8,
                                             kernel_size=3,
                                             strides=(2,  2),
                                             padding='valid',
                                             use_bias=False)

        # des_size = 47  # tf.shape(fused_cba2)[1]
        # src_size = 71  # tf.shape(deconv1)[1]
        #
        # src = t1asl_cba[:,
        #       tf.to_int32(src_size / 2) - tf.to_int32(des_size / 2) - 1:
        #       tf.to_int32(src_size / 2) + tf.to_int32(des_size / 2),
        #       tf.to_int32(src_size / 2) - tf.to_int32(des_size / 2) - 1:
        #       tf.to_int32(src_size / 2) + tf.to_int32(des_size / 2), :]
        conc2 = tf.concat([t1asl_cba, deconv2], -1)

        fused_cba6 = self.conv_batch_activation(variable_scope='feature_extractor_11_level_0', input=conc2,
                                                filters=8, kernel_size=3, padding='valid',
                                                is_training_bn=is_training_bn)
        tmp=tf.layers.conv2d(fused_cba6,
                         filters=8,
                         kernel_size=3,
                         padding='valid',
                         activation=None,
                         dilation_rate=3,
                         )
        with tf.variable_scope('last_layer'):
            conv_last = tf.layers.conv2d(tmp,
                                     filters=1,
                                     kernel_size=1,
                                     padding='valid',
                                     activation=None,
                                     dilation_rate=1,
                                     )

        y=conv_last
        return y,augmented_data