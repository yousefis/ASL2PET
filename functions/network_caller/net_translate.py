import shutil
# from functions.densenet_unet import _densenet_unet
# from functions.networks.dense_unet2 import _densenet_unet
import logging
from functions.cnn.multi_stage_denseunet import multi_stage_densenet
# import wandb
from functions.reader.data_reader import *
from functions.reader.image_class import *
from functions.losses.ssim_loss import multistage_SSIM,SSIM
from functions.threads import *
import psutil
# calculate the dice coefficient
from shutil import copyfile
from functions.reader.patch_extractor import _patch_extractor_thread
from functions.reader.data_reader import _read_data
# --------------------------------------------------------------------------------------------------------
class net_translate:
    def __init__(self,data_path,server_path , Logs,config):
        settings.init()
        self.data_path=data_path
        self.validation_samples=200
        self.Logs = Logs
        self.LOGDIR = server_path + self.Logs +  '/'
        self.learning_rate = .0001
        self.total_epochs=1000

        self.no_sample_per_each_itr = 1000
        self.sample_no = 2000000
        self.img_width = 500
        self.img_height = 500
        self.asl_size = 77 #input slice size
        self.pet_size = 63 #output slice size
        self.display_validation_step=5
        self.batch_no_validation=10
        self.batch_no=10
        self.parent_path = '/exports/lkeb-hpc/syousefi/Code/'
        self.chckpnt_dir = self.parent_path + self.Logs + '/unet_checkpoints/'
        self.config=config

    def copytree(self,src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
    def save_file(self,file_name,txt):
        with open(file_name, 'a') as file:
            file.write(txt)

    def count_number_trainable_params(self):
        '''
        Counts the number of trainable variables.
        '''
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(self,shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params
    def run_net(self):

        self.alpha_coeff=1

        '''read path of the images for train, test, and validation'''
        _rd = _read_data(self.data_path)
        train_data, validation_data, test_data=_rd.read_data_path()


        # ======================================
        bunch_of_images_no=1
        _image_class_vl = image_class(validation_data,
                                      bunch_of_images_no=bunch_of_images_no,
                                      is_training=0,inp_size=self.asl_size,out_size=self.pet_size)
        _patch_extractor_thread_vl = _patch_extractor_thread(_image_class=_image_class_vl,
                                                             img_no=bunch_of_images_no,
                                                             mutex=settings.mutex,
                                                             is_training=0,
                                                             )
        _fill_thread_vl = fill_thread(validation_data,
                                      _image_class_vl,
                                      mutex=settings.mutex,
                                      is_training=0,
                                      patch_extractor=_patch_extractor_thread_vl,
                                      )

        _read_thread_vl = read_thread(_fill_thread_vl, mutex=settings.mutex,
                                      validation_sample_no=self.validation_samples, is_training=0)
        _fill_thread_vl.start()
        _patch_extractor_thread_vl.start()
        _read_thread_vl.start()
        # ======================================
        bunch_of_images_no = 15
        _image_class_tr = image_class(train_data,
                                   bunch_of_images_no=bunch_of_images_no,
                                   is_training=1,inp_size=self.asl_size,out_size=self.pet_size
                                   )
        _patch_extractor_thread_tr = _patch_extractor_thread(_image_class=_image_class_tr,
                                                             img_no=bunch_of_images_no,
                                                             mutex=settings.mutex,
                                                             is_training=1,
                                                             )
        _fill_thread = fill_thread(train_data,
                                   _image_class_tr,
                                   mutex=settings.mutex,
                                   is_training=1,
                                   patch_extractor=_patch_extractor_thread_tr,
                                   )
        _read_thread = read_thread(_fill_thread, mutex=settings.mutex, is_training=1)
        _fill_thread.start()
        _patch_extractor_thread_tr.start()
        _read_thread.start()
        # ======================================
        # asl_plchld= tf.placeholder(tf.float32, shape=[None, None, None, 1])
        # t1_plchld= tf.placeholder(tf.float32, shape=[None, None, None, 1])
        # pet_plchld= tf.placeholder(tf.float32, shape=[None, None, None, 1])
        asl_plchld = tf.placeholder(tf.float32, shape=[self.batch_no, self.asl_size, self.asl_size, 1])
        t1_plchld = tf.placeholder(tf.float32, shape=[self.batch_no, self.asl_size, self.asl_size, 1])
        pet_plchld = tf.placeholder(tf.float32, shape=[self.batch_no, self.pet_size, self.pet_size, 1])


        ave_loss_vali = tf.placeholder(tf.float32)

        is_training = tf.placeholder(tf.bool, name='is_training')
        is_training_bn = tf.placeholder(tf.bool, name='is_training_bn')

        # cnn_net = unet()  # create object
        # y,augmented_data = cnn_net.unet(t1=t1_plchld, asl=asl_plchld, pet=pet_plchld, is_training_bn=is_training_bn)
        msdensnet = multi_stage_densenet()
        y,augmented_data,\
        level_ds1,level_ds2,level_ds3,level_us2,level_us3  = msdensnet.multi_stage_densenet(asl_img=asl_plchld,
                                                                                            t1_img=t1_plchld,
                                                                                              pet_img=pet_plchld,
                                                                                            input_dim=77,
                                                                                            is_training=is_training,
                                                                                              config=self.config)

        show_img = level_ds1[:, :, :, 0, np.newaxis]
        tf.summary.image('04: level_ds1', show_img, 10)

        show_img = level_ds2[:, :, :, 0, np.newaxis]
        tf.summary.image('05: level_ds2', show_img, 10)

        show_img = level_ds3[:, :, :, 0, np.newaxis]
        tf.summary.image('06: level_ds3', show_img, 10)

        show_img = level_us2[:, :, :, 0, np.newaxis]
        tf.summary.image('07: level_us2', show_img, 10)

        show_img = level_us3[:, :, :, 0, np.newaxis]
        tf.summary.image('08: level_us3', show_img, 10)

        show_img=augmented_data[0][:, :, :, 0, np.newaxis]
        tf.summary.image('00: input_asl', show_img, 3)

        show_img = augmented_data[1][:, :, :, 0, np.newaxis]
        tf.summary.image('01: input_t1', show_img, 3)

        show_img = augmented_data[2][:, :, :, 0, np.newaxis]
        tf.summary.image('02: target_pet', show_img, 3)
        #
        show_img = y[:, :, :, 0, np.newaxis]
        tf.summary.image('03: output_pet', show_img, 3)
#-----------------
        # show_img = loss_upsampling11[:, :, :, 0, np.newaxis]
        # tf.summary.image('04: loss_upsampling11', show_img, 3)
        # #
        # show_img = loss_upsampling22[:, :, :, 0, np.newaxis]
        # tf.summary.image('05: loss_upsampling22', show_img, 3)

        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # devices = sess.list_devices()
        # print(devices)
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')

        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' , graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' , graph=sess.graph)
        os.mkdir(self.LOGDIR + 'code/')
        copyfile('./run_net.py', self.LOGDIR + 'code/run_net.py')
        copyfile('./submit_job.py', self.LOGDIR + 'code/submit_job.py')
        copyfile('./test_file.py', self.LOGDIR + 'code/test_file.py')
        shutil.copytree('./functions/',self.LOGDIR+'code/functions/')

        # validation_writer.flush()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        # train_writer.close()
        # validation_writer.close()

        loadModel = 0
        # self.loss = ssim_loss()
        with tf.name_scope('cost'):
            ssim_val=SSIM(x1=pet_plchld, x2=y,max_val=2.1)[0]
            cost = tf.reduce_mean((1.0 - ssim_val), name="cost")
            # ssim_val=tf.reduce_mean(multistage_SSIM(x1=pet_plchld, x2=y,level1=loss_upsampling11, level2=loss_upsampling2,max_val=2.1))
            # cost = tf.reduce_mean((ssim_val), name="cost")
            # ssim_val = tf.reduce_mean(tf_ms_ssim(pet_plchld, y,level=3))
            # cost = tf.reduce_mean((ssim_val), name="cost")
            # mse=mean_squared_error(labels=augmented_data[-1],logit=y)

            # cost = tf.reduce_mean(mse , name="cost")
        tf.summary.scalar("cost", cost)
        # tf.summary.scalar("denominator", denominator)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, ).minimize(cost)

        with tf.name_scope('validation'):
            average_validation_loss = ave_loss_vali

        tf.summary.scalar("average_validation_loss", average_validation_loss)
        sess.run(tf.global_variables_initializer())




        logging.debug('total number of variables %s' % (
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        summ=tf.summary.merge_all()

        point = 0
        itr1 = 0
        if loadModel:
            chckpnt_dir=''
            ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            point=np.int16(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            itr1=point

        # with tf.Session() as sess:
        print("Number of trainable parameters: %d" % self.count_number_trainable_params())

        # patch_radius = 49
        '''loop for epochs'''

        
        for epoch in range(self.total_epochs):
            while self.no_sample_per_each_itr*int(point/self.no_sample_per_each_itr)<self.sample_no:
                print("epoch #: %d" %(epoch))
                startTime = time.time()
                step = 0
                self.beta_coeff=1+1 * np.exp(-point/2000)              

                # =============validation================
                if itr1 % self.display_validation_step ==0:
                    '''Validation: '''
                    loss_validation = 0
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation=0
              

                    while (validation_step * self.batch_no_validation < settings.validation_totalimg_patch):


                        [validation_asl_slices, validation_pet_slices,validation_t1_slices] = _image_class_vl.return_patches_validation( validation_step * self.batch_no_validation, (validation_step + 1) *self.batch_no_validation)
                        if (len(validation_asl_slices)<self.batch_no_validation) | (len(validation_pet_slices)<self.batch_no_validation) | (len(validation_t1_slices)<self.batch_no_validation) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            # print('sleep 3 validation')
                            continue


                        tic=time.time()
                        [loss_vali] = sess.run([ cost],
                                                 feed_dict={asl_plchld:validation_asl_slices ,
                                                            t1_plchld: validation_t1_slices,
                                                            pet_plchld: validation_pet_slices,
                                                            is_training: False,
                                                            ave_loss_vali: -1,
                                                            is_training_bn:False,
                                                           })
                        elapsed=time.time()-tic
                        loss_validation += loss_vali
                        validation_step += 1
                        if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                            print('nan problem')
                        process = psutil.Process(os.getpid())


                        print(
                            '%d - > %d: elapsed_time:%d  loss_validation: %f, memory_percent: %4s' % (
                                validation_step,validation_step * self.batch_no_validation
                                , elapsed,  loss_vali, str(process.memory_percent()),
                            ))

                            # end while
                    settings.queue_isready_vl = False
                    acc_validation = acc_validation / (validation_step)
                    loss_validation = loss_validation / (validation_step)
                    dsc_validation = dsc_validation / (validation_step)
                    if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                        print('nan problem')
                    _fill_thread_vl.kill_thread()
                    print('******Validation, step: %d , accuracy: %.4f, loss: %f*******' % (
                    itr1, acc_validation, loss_validation))

                    [sum_validation] = sess.run([summ],
                                           feed_dict={asl_plchld: validation_asl_slices,
                                                      t1_plchld: validation_t1_slices,
                                                      pet_plchld: validation_pet_slices,
                                                      is_training: False,
                                                      ave_loss_vali: loss_validation,
                                                      is_training_bn: False,
                                                      })

                    validation_writer.add_summary(sum_validation, point)
                    validation_writer.flush()
                    print('end of validation---------%d' % (point))
                    # end if
                '''loop for training batches'''

                while(step*self.batch_no<self.no_sample_per_each_itr):

                    [train_asl_slices,train_pet_slices,train_t1_slices] = _image_class_tr.return_patches( self.batch_no)



                    if (len(train_asl_slices)<self.batch_no)|(len(train_pet_slices)<self.batch_no)\
                            |(len(train_t1_slices)<self.batch_no):
                        #|(len(train_t1_slices)<self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue


                    tic=time.time()

                    [ loss_train1,opt,] = sess.run([ cost,optimizer,],
                                             feed_dict={asl_plchld: train_asl_slices,
                                                        t1_plchld: train_t1_slices,
                                                        pet_plchld: train_pet_slices,
                                                        is_training: True,
                                                        ave_loss_vali: -1,
                                                        is_training_bn: True})
                    elapsed=time.time()-tic
                    [sum_train] = sess.run([summ],
                                           feed_dict={asl_plchld: train_asl_slices,
                                                        t1_plchld: train_t1_slices,
                                                        pet_plchld: train_pet_slices,
                                                        is_training: False,
                                                        ave_loss_vali: loss_train1,
                                                        is_training_bn: False
                                                      })
                    train_writer.add_summary(sum_train,point)
                    train_writer.flush()
                    step = step + 1

                    process = psutil.Process(os.getpid())

                    print(
                        'point: %d, elapsed_time:%d step*self.batch_no:%f , LR: %.15f, loss_train1:%f,memory_percent: %4s' % (
                        int((point)),elapsed,
                        step * self.batch_no, self.learning_rate, loss_train1,
                        str(process.memory_percent())))


                    point=int((point))#(self.no_sample_per_each_itr/self.batch_no)*itr1+step

                    if point%100==0:
                        '''saveing model inter epoch'''
                        chckpnt_path = os.path.join(self.chckpnt_dir,
                                                    ('densenet_unet_inter_epoch%d_point%d.ckpt' % (epoch, point)))
                        saver.save(sess, chckpnt_path, global_step=point)



                    itr1 = itr1 + 1
                    point=point+1







            endTime = time.time()




            #==============end of epoch:



            '''saveing model after each epoch'''
            chckpnt_path = os.path.join(self.chckpnt_dir, 'densenet_unet.ckpt')
            saver.save(sess, chckpnt_path, global_step=epoch)
            print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime - startTime))
