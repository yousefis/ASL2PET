import tensorflow as tf

class Models:
    def __init__(self, param):
        self.network_type = param.network_type
        self.learning_type = param.learning_type
        self.weight_decay = param.weight_decay

        if param.data_type == 'float32':
            self.data_type = tf.float32
        elif param.data_type == 'float16':
            self.data_type = tf.float16

        if param.regularizer == 'l1':
            self.regularizer = tf.keras.regularizers.l1(l=self.weight_decay)
        elif param.regularizer == 'l2':
            self.regularizer = tf.keras.regularizers.l2(l=self.weight_decay)

        if param.initializer == 'xavier':
            self.initializer = tf.keras.initializers.glorot_normal()
        elif param.initializer == 'he':
            self.initializer = tf.keras.initializers.he_normal()
        elif param.initializer == 'gaussian':
            self.initializer = tf.initializers.truncated_normal(stddev=0.02, dtype=self.data_type)

        self.input_y_size = param.input_y_size
        self.input_x_size = param.input_x_size
        self.input_ch_size = param.input_ch_size

        if param.patch_y_size == -1:
            self.patch_y_size = param.input_y_size
        else:
            self.patch_y_size = param.patch_y_size

        if param.patch_x_size == -1:
            self.patch_x_size = param.input_x_size
        else:
            self.patch_x_size = param.patch_x_size

        if param.patch_ch_size == -1:
            self.patch_ch_size = param.input_ch_size
        else:
            self.patch_ch_size = param.patch_ch_size

    '''
    SELECT NETWORK STRUCTURE such as RES-NET and U-NET
    '''
    def get_network(self, input, training, network_type='unet', reuse=tf.AUTO_REUSE, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            if network_type == 'unet':
                output = self.get_unet(input=input, training=training)
            elif network_type == 'autoencoder':
                output = self.get_autoencoder(input=input, training=training)
            elif network_type == 'resnet':
                output = self.get_resnet(input=input, training=training)

        return output

    def get_vars(self, name='generator'):
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if name in var.name]

        return vars

    '''
    RES-NET STRUCTURE
    '''
    def get_resnet(self, input, training):
        print('currently, it dose not implemented.')
        return []

    '''
    AUTOENCODER STRUCTURE
    '''
    def get_autoencoder(self, input, training):

        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, training=training, filters=[64, 64], is_bnorm=True, is_activate=True)
            pool0 = self.get_pool2d(enc0)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(pool0, training=training, filters=[128, 128], is_bnorm=True, is_activate=True)
            pool1 = self.get_pool2d(enc1)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(pool1, training=training, filters=[256, 256], is_bnorm=True, is_activate=True)
            pool2 = self.get_pool2d(enc2)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(pool2, training=training, filters=[512, 512], is_bnorm=True, is_activate=True)
            pool3 = self.get_pool2d(enc3)

        '''
        ENCODER & DECODER PART
        '''
        stg = 4
        with tf.variable_scope('enc_dec_{}'.format(stg)):
            enc4 = self.get_standard_block(pool3, training=training, filters=[1024, 512], is_bnorm=True, is_activate=True)

        '''
        DECODER PART
        '''
        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool3 = self.get_unpool2d(enc4)
            # concat3 = self.get_concat([unpool3, enc3])
            dec3 = self.get_standard_block(unpool3, training=training, filters=[512, 256], is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool2 = self.get_unpool2d(dec3)
            # concat2 = self.get_concat([unpool2, enc2])
            dec2 = self.get_standard_block(unpool2, training=training, filters=[256, 128], is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool1 = self.get_unpool2d(dec2)
            # concat1 = self.get_concat([unpool1, enc1])
            dec1 = self.get_standard_block(unpool1, training=training, filters=[128, 64], is_bnorm=True, is_activate=True)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool0 = self.get_unpool2d(dec1)
            # concat0 = self.get_concat([unpool0, enc0])
            dec0 = self.get_standard_block(unpool0, training=training, filters=[64, 64], is_bnorm=True, is_activate=True)

        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('fc'):
            output = self.get_standard_block(dec0, filters=[self.patch_ch_size], kernel_size=[1, 1], is_bnorm=False, is_activate=False)

        if self.learning_type == 'residual':
            output = tf.add(output, input)

        return output

    '''
    U-NET STRUCTURE
    '''
    def get_unet(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, training=training, filters=[64, 64], is_bnorm=True, is_activate=True)
            pool0 = self.get_pool2d(enc0)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(pool0, training=training, filters=[128, 128], is_bnorm=True, is_activate=True)
            pool1 = self.get_pool2d(enc1)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(pool1, training=training, filters=[256, 256], is_bnorm=True, is_activate=True)
            pool2 = self.get_pool2d(enc2)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(pool2, training=training, filters=[512, 512], is_bnorm=True, is_activate=True)
            pool3 = self.get_pool2d(enc3)

        '''
        ENCODER & DECODER PART
        '''
        stg = 4
        with tf.variable_scope('enc_dec_{}'.format(stg)):
            enc4 = self.get_standard_block(pool3, training=training, filters=[1024, 512], is_bnorm=True, is_activate=True)

        '''
        DECODER PART
        '''
        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool3 = self.get_unpool2d(enc4)
            concat3 = self.get_concat([unpool3, enc3])
            dec3 = self.get_standard_block(concat3, training=training, filters=[512, 256], is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool2 = self.get_unpool2d(dec3)
            concat2 = self.get_concat([unpool2, enc2])
            dec2 = self.get_standard_block(concat2, training=training, filters=[256, 128], is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool1 = self.get_unpool2d(dec2)
            concat1 = self.get_concat([unpool1, enc1])
            dec1 = self.get_standard_block(concat1, training=training, filters=[128, 64], is_bnorm=True, is_activate=True)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool0 = self.get_unpool2d(dec1)
            concat0 = self.get_concat([unpool0, enc0])
            dec0 = self.get_standard_block(concat0, training=training, filters=[64, 64], is_bnorm=True, is_activate=True)

        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('fc'):
            output = self.get_standard_block(dec0, filters=[self.patch_ch_size], kernel_size=[1, 1], is_bnorm=False, is_activate=False)

        if self.learning_type == 'residual':
            with tf.variable_scope('add'):
                output = tf.add(output, input)

        return output

    '''
    GET STANDARD BLOCK: CONV -> BNORM -> RELU 
    '''
    def get_standard_block(self, input, filters, kernel_size=[3, 3], stride=1, padding='same', training=True, is_bnorm=True, is_activate=True):
        output = input

        for i, f in enumerate(filters):
            with tf.variable_scope('conv_{}'.format(i)):
                output = tf.layers.conv2d(output,
                                          filters=f,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          activation=None)
            if is_bnorm:
                with tf.variable_scope('bnorm_{}'.format(i)):
                    output = tf.layers.batch_normalization(output,
                                                           training=training)
            if is_activate:
                with tf.variable_scope('relu_{}'.format(i)):
                    output = tf.nn.relu(output)

        return output


    def get_residual_block(self, input, filters, kernel_size=[3, 3], stride=1, padding='same', training=True, is_bnorm=True, is_activate=True):
        output = input

        for i, f in enumerate(filters):
            with tf.variable_scope('conv_{}'.format(i)):
                output = tf.layers.conv2d(output,
                                          filters=f,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          activation=None)
            if is_bnorm:
                with tf.variable_scope('bnorm_{}'.format(i)):
                    output = tf.layers.batch_normalization(output,
                                                           training=training)

            if i == len(filters) - 1:
                with tf.variable_scope('add'.format(i)):
                    output = tf.add(output, input)

            if is_activate:
                with tf.variable_scope('relu_{}'.format(i)):
                    output = tf.nn.relu(output)

        return output


    '''
    GET SQUENTIAL CONV-BNORM-RELU LAYERS
    '''
    def get_conv2d(self, input, filters, kernel_size=[3, 3], stride=1, padding='same', name='conv'):
        with tf.variable_scope(name):
            output = tf.layers.conv2d(input,
                                      filters=filters,
                                      kernel_size=kernel_size,
                                      strides=stride,
                                      padding=padding,
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=self.regularizer,
                                      activation=None)
        return output

    def get_bnorm2d(self, input, training=True, name='bnorm'):
        with tf.variable_scope(name):
            output = tf.layers.batch_normalization(input,
                                                   training=training)
        return output

    def get_relu(self, input, name='relu'):
        with tf.variable_scope(name):
            output = tf.nn.relu(input)
            # output = tf.maximum(input, 0)
        return output

    def get_lrelu(self, input, leak=0.2, name='lrelu'):
        with tf.variable_scope(name):
            output = tf.maximum(input, leak*input)
        return output

    '''
    GET POOLING LAYERS such as AVERAGE POOLING and MAX POOLING
    '''
    def get_pool2d(self, input, pool_size=2, strides=2, padding='same', type='avg', name='pool_{}'):
        with tf.variable_scope(name.format(type)):
            if type == 'avg':
                output = tf.layers.average_pooling2d(input,
                                                     pool_size=pool_size,
                                                     strides=strides,
                                                     padding=padding)
            elif type == 'max':
                output = tf.layers.max_pooling2d(input,
                                                 pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding)
        return output

    '''
    GET UN-POOLING LAYERS such as AVERAGE POOLING and UNSAMPLING using CONVT LAYER
    '''
    def get_unpool2d(self, input, pool_size=2, strides=2, padding='same', type='avg', name='unpool_{}'):
        with tf.variable_scope(name.format(type)):
            if type == 'avg':
                output = tf.cast(tf.image.resize_nearest_neighbor(input,
                                                                  size=[input.shape.dims[1] * pool_size,
                                                                        input.shape.dims[2] * pool_size]),
                                 dtype=self.data_type)
            elif type == 'convt':
                output = tf.layers.conv2d_transpose(input,
                                                    filters=input.shape.dims[3],
                                                    kernel_size=strides,
                                                    strides=strides,
                                                    padding=padding,
                                                    kernel_initializer=self.initializer,
                                                    kernel_regularizer=self.regularizer)
        return output

    '''
    GET CONCAT LAYERS
    '''
    def get_concat(self, inputs, axis=3):
        output = tf.concat(inputs, axis=axis)
        return output

    '''
    GET FFT LAYERS
    '''
    def get_fft(self, inputs):
        # H W C B
        output = tf.transpose(inputs, perm=[1, 2, 3, 0])
        sz = tf.shape(inputs)

        # [REAL, IMAG] to COMPLEX
        output = tf.complex(output[:, :, 0:sz[2]/2, :], output[:, :, sz[2]/2:, :])

        # IFFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.ceil(sz[0]/2), tf.ceil(sz[1]/2)], axis=[0, 1])
        # FFT: 2D
        output = tf.spectral.fft2d(output)
        # FFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.floor(sz[0]/2), tf.floor(sz[1]/2)], axis=[0, 1])

        # COMPLEX to [REAL, IMAG]
        output = tf.concat([tf.real(output), tf.imag(output)], axis=2)

        # B H W C
        output = tf.transpose(output, perm=[3, 0, 1, 2])

        return output

    '''
    GET IFFT LAYERS
    '''
    def get_ifft(self, inputs):
        # H W C B
        output = tf.transpose(inputs, perm=[1, 2, 3, 0])
        sz = tf.shape(inputs)

        # [REAL, IMAG] to COMPLEX
        output = tf.complex(output[:, :, :(sz[2]/2), :], output[:, :, (sz[2]/2):, :])

        # FFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.floor(sz[0] / 2), tf.floor(sz[1] / 2)], axis=[0, 1])
        # IFFT: 2D
        output = tf.spectral.ifft2d(output)
        # IFFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.ceil(sz[0] / 2), tf.ceil(sz[1] / 2)], axis=[0, 1])

        # COMPLEX to [REAL, IMAG]
        output = tf.concat([tf.real(output), tf.imag(output)], axis=2)

        # B H W C
        output = tf.transpose(output, perm=[3, 0, 1, 2])

        return output