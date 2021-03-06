#do test process
import matplotlib.pyplot as plt
import matplotlib

import time
from shutil import copyfile
from functions.losses.ssim_loss import SSIM
from functions.reader.data_reader import *
from functions.reader.data_reader import _read_data
from functions.cnn.multi_stage_denseunet import multi_stage_densenet
from functions.measures.measures import _measure
from functions.reader.image_class import image_class
import pandas as pd
import numpy as np
import SimpleITK as sitk
eps = 1E-5
plot_tag = 'densenet'
config =  [2, 3, 4, 3, 2]
pet_size = 63
asl_size = 77
with tf.variable_scope('crop_claculation'):
    in_size0 = tf.to_int32(0, name='in_size0')
    in_size1 = tf.to_int32(asl_size, name='in_size1')
    in_size2 = tf.to_int32(in_size1, name='in_size2')  # conv stack
    in_size3 = tf.to_int32((in_size2 - 2), name='in_size3')  # level_design1
    in_size4 = tf.to_int32(in_size3 / 2 - 2, name='in_size4')  # downsampleing1+level_design2
    in_size5 = tf.to_int32(in_size4 / 2 - 2, name='in_size5')  # downsampleing2+level_design3
    crop_size0 = tf.to_int32(0, name='crop_size0')
    crop_size1 = tf.to_int32(2 * in_size5 + 1, name='crop_size1')
    crop_size2 = tf.to_int32(2 * crop_size1 + 1, name='crop_size2')
    




gap = asl_size - pet_size





def test_all_nets(out_dir,Log,which_data):


   
    
    data_path = "/exports/lkeb-hpc/syousefi/Data/asl_pet/"
    _rd = _read_data(data_path)

    train_data, validation_data, test_data = _rd.read_data_path()

    if which_data==1:
        data = validation_data
    elif which_data==2:
        data = test_data
    elif which_data == 3:
        data = train_data


    asl_plchld = tf.placeholder(tf.float32, shape=[None, asl_size, asl_size, 1])
    t1_plchld = tf.placeholder(tf.float32, shape=[None, asl_size, asl_size, 1])
    pet_plchld = tf.placeholder(tf.float32, shape=[None, pet_size, pet_size, 1])

    ave_loss_vali = tf.placeholder(tf.float32)

    is_training = tf.placeholder(tf.bool, name='is_training')
    is_training_bn = tf.placeholder(tf.bool, name='is_training_bn')

    msdensnet = multi_stage_densenet()
    y, augmented_data, \
    level_ds1, level_ds2, level_ds3, level_us2, level_us3 = msdensnet.multi_stage_densenet(asl_img=asl_plchld,
                                                                                           t1_img=t1_plchld,
                                                                                           pet_img=pet_plchld,
                                                                                           input_dim=77,
                                                                                           is_training=is_training,
                                                                                           config=config)

    with tf.name_scope('cost'):
        ssim_val = SSIM(x1=pet_plchld, x2=y, max_val=2.1)[0]
        cost = tf.reduce_mean((1.0 - ssim_val), name="cost")
        
    sess = tf.Session()
    saver=tf.train.Saver()
    parent_path='/exports/lkeb-hpc/syousefi/Code/'
    chckpnt_dir=parent_path+Log+'unet_checkpoints/'
    ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    _meas = _measure()
    copyfile('./test_net.py',
             parent_path+ Log + out_dir + 'test_net.py')

    _image_class = image_class(data,
                                  bunch_of_images_no=1,
                                  is_training=0,
                                  inp_size=asl_size,
                                  out_size=pet_size)
    list_ssim=[]
    list_name=[]
    for scan in range(len(data)):
        ss = str(data[scan]['asl']).split("/")
        imm = _image_class.read_image(data[scan])
        try:
            os.mkdir(parent_path + Log + out_dir +ss[-3])
        except:
            a=1
        os.mkdir(parent_path + Log + out_dir +ss[-3]+'/'+ss[-1].split(".")[0].split("ASL_")[1])
        for img_indx in range(np.shape(imm[3])[0]):
            print('img_indx:%s' %(img_indx))

            name =ss[-3] + '_' +ss[-2] + '_' +str(img_indx)
            # name = (ss[10] + '_' + ss[11] + '_' + ss[12].split('%')[0]).split('_CT')[0]
            tic = time.time()
            t1 = imm[3][img_indx,
                         40-int(asl_size/2)-1:40+int(asl_size/2),
                         40-int(asl_size/2)-1:40+int(asl_size/2)]
            t1 = t1[np.newaxis,...,np.newaxis]
            asl = imm[4][np.newaxis,img_indx,
                         40-int(asl_size/2)-1:40+int(asl_size/2),
                         40-int(asl_size/2)-1:40+int(asl_size/2),np.newaxis]
            pet = imm[5][np.newaxis,img_indx,
                          40-int(pet_size/2)-1:40+int(pet_size/2),
                          40-int(pet_size/2)-1:40+int(pet_size/2),np.newaxis]
            [loss, out, ] = sess.run([cost,y ],
                                            feed_dict={asl_plchld: asl,
                                                       t1_plchld: t1,
                                                       pet_plchld: pet,
                                                       is_training: False,
                                                       ave_loss_vali: -1,
                                                       is_training_bn: False})
            
            # plt.imshow(np.squeeze(out))
            # plt.figure()
            # plt.imshow(np.squeeze(pet))
            ssim=1-loss
            list_ssim.append(ssim)
            list_name.append(ss[-3]+'_'+ss[-1].split(".")[0].split("ASL_")[1] + '_t1_'+name)
            print(ssim)
            matplotlib.image.imsave(parent_path + Log + out_dir+ss[-3]+'/'+ss[-1].split(".")[0].split("ASL_")[1] + '/t1_'+name+'.png', np.squeeze(t1), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir +ss[-3]+'/'+ss[-1].split(".")[0].split("ASL_")[1]+ '/asl_'+name+'_'+'.png', np.squeeze(asl), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir +ss[-3]+'/'+ss[-1].split(".")[0].split("ASL_")[1]+ '/pet_'+name+'_'+'.png', np.squeeze(pet), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir +ss[-3]+'/'+ ss[-1].split(".")[0].split("ASL_")[1]+'/res_'+name+'_'+str(ssim)+'.png', np.squeeze(out), cmap='gray')

            elapsed = time.time() - tic


    df = pd.DataFrame(list_ssim,
                     columns=pd.Index(['ssim'],
                     name='Genus')).round(2)
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(parent_path + Log + out_dir+'/all_ssim.xlsx',
                            engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    print(parent_path + Log + out_dir + '/all_ssim.xlsx')

if __name__=="__main__":
    Log="Log_asl_pet/denseunet_multistage_mssim7/"
    which_data = 2
    if which_data==1:
        out_dir="0_vali_result/"
    elif which_data==2:
        out_dir="0_test_result/"
    elif which_data==3:
        out_dir="0_train_result/"

    test_all_nets( out_dir, Log,which_data=which_data)