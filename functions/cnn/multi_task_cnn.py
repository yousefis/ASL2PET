import math
import tensorflow as tf
# import math as math
import numpy as np
# import matplotlib.pyplot as plt
from functions.layers.upsampling import upsampling

# !!

class multi_task_cnn:
    def __init__(self, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed_no=200
        self.upsampling3d=upsampling()
    def seed(self):
        self.seed_no+=1
        return self.seed_no

    # ========================
    def convolution_stack(self, input, stack_name, filters, kernel_size, padding, is_training):
        with tf.variable_scope(stack_name):
            conv1 = tf.layers.conv2d(input,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        activation=None,
                                        dilation_rate=1,
                                        )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv1 = bn

            # conv2 = tf.layers.conv2d(conv1,
            #                          filters=filters,
            #                          kernel_size=kernel_size,
            #                          padding=padding,
            #                          activation=None,
            #                          dilation_rate=1)
            # bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            # bn = tf.nn.leaky_relu(bn)
            # conv2 = bn
            conc=tf.concat([input[:,1:-1,1:-1,:],conv1],4)
            # conc=tf.concat([conc,conv2],4)

            # conv3 = tf.layers.conv2d(conc,
            #                          filters=filters,
            #                          kernel_size=1,
            #                          padding=padding,
            #                          activation=None,
            #                          dilation_rate=1)
            # bn = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            # bn = tf.nn.leaky_relu(bn)
            # conv3 = bn
            return conc
    #==================================================
    def level_design(self,input,level_name,filters1,filters2,is_training,kernel_size,in_size,crop_size,padding1,padding2,flag=2,paddingfree_scope='',filters3=0):
        with tf.variable_scope(level_name):
            conv1 = tf.layers.conv2d(input,
                                     filters=filters1,
                                     kernel_size=kernel_size,
                                     padding=padding1,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv1 = bn
            conc = tf.concat([input, conv1], -1)

            conv2 = tf.layers.conv2d(conc,
                                     filters=filters2,
                                     kernel_size=kernel_size,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv2 = bn


            conc=tf.concat([conc,conv2],-1)


            # bottleneck layer
            conv3 = tf.layers.conv2d(conc,
                                     filters=filters2,
                                     kernel_size=1,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv3 = bn
            if flag==1:
                with tf.variable_scope(paddingfree_scope):
                    conc = self.paddingfree_conv(input=conv3, filters=filters3, kernel_size=3, is_training=is_training)
                crop = conc[:,
                          tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                          tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                          tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                          tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),:]
            if flag==2:
                crop = conc[:,
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]

            return conc,crop

    #==================================================
    def noisy_input(self,img_rows,is_training):
        noisy_img_rows=[]
        #
        with tf.variable_scope("Noise"):
            rnd = tf.greater_equal(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed()),5)[0]

            mean=  tf.random_uniform([1], maxval=5, seed=self.seed())
            stdev=  tf.random_uniform([1], maxval=7, seed=self.seed())

            for i in range(len(img_rows)):
                noisy_img_rows.append( tf.cond(tf.logical_and(is_training,rnd),
                                         lambda: img_rows[i] + tf.round(tf.random_normal(tf.shape(img_rows[i]),
                                                                                         mean=mean,
                                                                                         stddev=stdev,
                                                                                         seed=self.seed() ,
                                                                                         dtype=tf.float32))
                                         , lambda: img_rows[i]))

        return noisy_img_rows


    # ==================================================

    def rotate_input(self,img_rows,is_training):
        rotate_img_rows=[]
        #
        with tf.variable_scope("Rotate"):
            rnd = tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed()),4)[0] # , seed=int(time.time())))
            degree_angle = tf.random_uniform([1],minval=-45, maxval=45, seed=self.seed())[0]
            radian = degree_angle * math.pi / 180
            # if rnd:
            for i in range(len(img_rows)):
                rotate_img_rows.append( tf.cond(tf.logical_and(is_training,rnd),
                                          lambda: tf.expand_dims(tf.contrib.image.rotate(tf.squeeze(img_rows[i], 3), radian),
                                                                 axis=-1)
                                          , lambda: img_rows[i]))

        return rotate_img_rows,degree_angle
    # ========================

    def flip_lr_input(self, img_rows,is_training):
        flip_lr_img_rows=[]
        with tf.variable_scope("LR_Flip"):
            rnd =(tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed()),4))[0]  # , seed=int(time.time())))
            for i in range(len(img_rows)):
                flip_lr_img_rows.append( tf.cond(tf.logical_and(is_training, rnd),
                                           lambda: tf.expand_dims(tf.image.flip_up_down(tf.squeeze(img_rows[i], 3)),
                                                                  axis=-1)
                                           , lambda: img_rows[i]))


        return flip_lr_img_rows
    #==================================================
    def paddingfree_conv(self,input,filters,kernel_size,is_training):
        conv = tf.layers.conv2d(input,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding='valid',
                                 activation=None,
                                 dilation_rate=1,
                                 )
        bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
        bn = tf.nn.leaky_relu(bn)
        conv = bn
        return conv
    #==================================================
    def bspline3D(self,x, y, z):
        x = abs(x)
        y = abs(y)
        z = abs(z)
        if (((x + y + z) >= 0) and ((x + y + z) < 1)):
            f = (0.5) * ((x + y + z) ** 3) - ((x + y + z) ** 2) + 4 / 6
        elif ((x + y + z) >= 1 and (x + y + z) <= 2):
            f = (-1 / 6) * ((x + y + z) ** 3) + ((x + y + z) ** 2) - 2 * (x + y + z) + 8 / 6
        else:
            f = 0
        return f

    #==================================================
    def convDownsampleKernel(self,kernelName, dimension, kernelSize, normalizeKernel=None):
        numOfPoints = kernelSize + 2
        XInput = np.linspace(-2, 2, num=numOfPoints)

        if dimension == 3:
            YInput = np.linspace(-2, 2, num=numOfPoints)
            ZInput = np.linspace(-2, 2, num=numOfPoints)
            xv, yv, zv = np.meshgrid(XInput, YInput, ZInput)

            if kernelName == 'bspline':
                Y = np.stack(
                    [self.bspline3D(xv[i, j, k], yv[i, j, k], zv[i, j, k]) for i in range(0, np.shape(xv)[0]) for j in
                     range(0, np.shape(xv)[0])
                     for k in range(0, np.shape(xv)[0])])
            Y = np.reshape(Y, [len(XInput), len(XInput), len(XInput)])
            Y = Y[1:-1, 1:-1, 1:-1]
        if normalizeKernel:
            if np.sum(Y) != normalizeKernel:
                ratio = normalizeKernel / np.sum(Y)
                Y = ratio * Y

        Y[abs(Y) < 1e-6] = 0
        return Y.astype(np.float32)





    #==================================================
    def multi_stage_densenet(self, asl_img, t1_img,pet_img,  input_dim, is_training,conv_transpose=True):
        # mri=None
        # in_size0 = tf.to_int32(0, name='in_size0')
        # in_size1 = 77
        # in_size2 = np.int8(in_size1)  # conv stack
        # in_size3 = np.int8((in_size2 - 2))  # level_design1
        # in_size4 = np.int8(in_size3 / 2 - 2)  # downsampleing1+level_design2
        # in_size5 = np.int8(in_size4 / 2 - 2)  # downsampleing2+level_design3
        # crop_size0 = np.int8(0)
        # crop_size1 = np.int8(2 * in_size5 + 1,)
        # crop_size2 = np.int8(2 * crop_size1 + 1)

        # input_dim=77
        with tf.variable_scope('crop_claculation'):
            in_size0 = tf.to_int32(0,name='in_size0')
            in_size1 = tf.to_int32(input_dim,name='in_size1')
            in_size2 = tf.to_int32(in_size1 ,name='in_size2')  # conv stack
            in_size3 = tf.to_int32((in_size2-2),name='in_size3')  # level_design1
            in_size4 = tf.to_int32(in_size3 / 2-2 ,name='in_size4')  # downsampleing1+level_design2
            in_size5 = tf.to_int32(in_size4 / 2 -2,name='in_size5')  # downsampleing2+level_design3
            crop_size0 = tf.to_int32(0,name='crop_size0')
            crop_size1 = tf.to_int32(2 * in_size5 + 1,name='crop_size1')
            crop_size2 = tf.to_int32(2 * crop_size1  + 1,name='crop_size2')

        # asl_img = tf.layers.batch_normalization(asl_img, training=is_training, renorm=False)
        # t1_img = tf.layers.batch_normalization(t1_img, training=is_training, renorm=False)
        # pet_img = tf.layers.batch_normalization(pet_img, training=is_training, renorm=False)

        img_rows=[]
        img_rows.append(asl_img)
        img_rows.append(t1_img)
        img_rows.append(pet_img)



        # if mri !=None:
        #     img_rows.append(mri)

        with tf.variable_scope('augmentation'):
            with tf.variable_scope('noise'):
                img_rows1=self.noisy_input( img_rows[0:-1],is_training)
                img_rows1.append(img_rows[-1])
                img_rows=img_rows1
            with tf.variable_scope('LR_flip'):
                img_rows=self.flip_lr_input(img_rows, is_training)
            with tf.variable_scope('rotate'):
                img_rows,degree=self.rotate_input(img_rows, is_training)
        augmented_data=img_rows

        with tf.variable_scope('stack-contact'):
                stack_concat = tf.concat([img_rows[0], img_rows[1]], -1)



            # stack_concat = tf.concat([stack_concat, img_rows[3]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[4]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[5]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[6]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[7]], 4)
        #== == == == == == == == == == == == == == == == == == == == == ==
        #level 1 of unet
        [level_ds1, crop1] = self.level_design(stack_concat,
                                               'level_ds1',
                                               filters1=16,
                                               filters2=16,
                                               filters3=16,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv1',
                                               flag=1)
        with tf.variable_scope('maxpool1'):
            pool1 = tf.layers.max_pooling2d(inputs=level_ds1, pool_size=(2, 2), strides=(2, 2))
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1,
                                               'level_ds2',
                                               filters1=32,
                                               filters2=32,
                                               filters3=32,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv2',
                                               flag=1)
        with tf.variable_scope('maxpool2'):
            pool2 = tf.layers.max_pooling2d(inputs=level_ds2, pool_size=(2, 2), strides=(2, 2))

        # level 3 of unet
        [level_ds3, crop3] = self.level_design(pool2, 'level_ds3',
                                               filters1=64,
                                               filters2=64,
                                               filters3=64,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size5,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv3',
                                               flag=1)
        with tf.variable_scope('loss_upsampling1'):
            if conv_transpose:
                loss_upsampling1 = tf.layers.conv2d_transpose(level_ds3,
                                                              filters=1,
                                                              kernel_size=3,
                                                              strides=(2, 2),
                                                              padding='valid',
                                                              use_bias=False)
                loss_upsampling11 = tf.layers.conv2d_transpose(loss_upsampling1,
                                                               filters=1,
                                                               kernel_size=3,
                                                               strides=(2, 2),
                                                               padding='valid',
                                                               use_bias=False)
            else:
                conv_up1 = tf.layers.conv2d(level_ds3,
                                         filters=1,
                                         kernel_size=3,
                                         padding='same',
                                         activation=None,
                                         dilation_rate=1,
                                         )
                bn = tf.layers.batch_normalization(conv_up1, training=is_training, renorm=False)
                bn = tf.nn.leaky_relu(bn)
                loss_upsampling1 = self.upsampling3d.upsampling2d(bn,
                                                                  scope='loss_upsampling1',
                                                                  scale=2,
                                                                  interpolator='trilinear',
                                                                  padding_mode='SYMMETRIC',
                                                                  padding_constant=0,
                                                                  trainable=False,
                                                                  padding='valid')
                loss_upsampling11 = self.upsampling3d.upsampling2d(loss_upsampling1,
                                                                   scope='loss_upsampling11',
                                                                   scale=2,
                                                                   interpolator='trilinear',
                                                                   padding_mode='SYMMETRIC',
                                                                   padding_constant=0,
                                                                   trainable=False,
                                                                   padding='valid')

        with tf.variable_scope('conv_transpose1'):
            deconv1 = tf.layers.conv2d_transpose(level_ds3,
                                                 filters=32,
                                                 kernel_size=3,
                                                 strides=(2, 2),
                                                 padding='valid',
                                                 use_bias=False)

        with tf.variable_scope('concat1'):
            conc12 = tf.concat([crop2, deconv1], -1)

        # level 2 of unet
        [level_us2, crop0] = self.level_design(conc12, 'level_us2',
                                               filters1=32,
                                               filters2=32,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')
        with tf.variable_scope('loss_upsampling2'):
            if conv_transpose:
                loss_upsampling2 = tf.layers.conv2d_transpose(level_us2,
                                                     filters=1,
                                                     kernel_size=3,
                                                     strides=(2, 2),
                                                     padding='valid',
                                                     use_bias=False)
            else:
                conv_up2 = tf.layers.conv2d(level_us2,
                                         filters=1,
                                         kernel_size=3,
                                         padding='same',
                                         activation=None,
                                         dilation_rate=1,
                                         )
                bn = tf.layers.batch_normalization(conv_up2, training=is_training, renorm=False)
                bn = tf.nn.leaky_relu(bn)
                loss_upsampling2 = self.upsampling3d.upsampling2d(bn,
                                                                  scope='loss_upsampling2',
                                                                  scale=2,
                                                                  interpolator='trilinear',
                                                                  padding_mode='SYMMETRIC',
                                                                  padding_constant=0,
                                                                  trainable=False,
                                                                  padding='valid')

        with tf.variable_scope('conv_transpose2'):
            deconv2 = tf.layers.conv2d_transpose(level_us2,
                                                 filters=16,
                                                 kernel_size=3,
                                                 strides=(2, 2),
                                                 padding='valid',
                                                 use_bias=False)

        with tf.variable_scope('concat2'):
            conc23 = tf.concat([crop1, deconv2], -1)

        # level 1 of unet
        [level_us3, crop0] = self.level_design(conc23, 'level_us3',
                                               filters1=16,
                                               filters2=16,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')

        with tf.variable_scope('last_layer'):
            conv1 = tf.layers.conv2d(level_us3,
                                     filters=8,
                                     kernel_size=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            y = tf.nn.leaky_relu(bn)
            y= tf.layers.conv2d(y,
                             filters=1,
                             kernel_size=1,
                             padding='same',
                             activation=None,
                             dilation_rate=1,
                             )



        # == == == == == == == == == == == == == == == == == == == == == ==


        # classification layer:
        # with tf.variable_scope('classification_layer'):
        #     y = tf.layers.conv2d(bn, filters=self.class_no, kernel_size=14, padding='same', strides=(1, 1, 1),
        #                          activation=None, dilation_rate=(1, 1,1), name='fc3'+self.log_ext)

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))





        return  y,augmented_data,loss_upsampling11,loss_upsampling2
