# do test process
import matplotlib.pyplot as plt
import matplotlib

import time
from shutil import copyfile
from functions.losses.ssim_loss import SSIM
from functions.reader.data_reader_hybrid_hr import *
from functions.reader.data_reader_hybrid_hr import _read_data
# from functions.cnn.hybrid_multi_task_cnn import multi_stage_densenet
from functions.cnn.hybrid_multi_task_cnn_hr_not1 import multi_stage_densenet
from functions.measures.measures import _measure
from functions.reader.image_class_hybrid_hr import image_class



# import wandb


import pandas as pd
import numpy as np
import SimpleITK as sitk
from functions.losses.L1 import huber

eps = 1E-5
plot_tag = 'densenet'
config = [1, 3, 5, 3, 1]

asl_size =231
pet_size = 219
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


def test_all_nets(out_dir, Log, which_data):
    data_path_AMUC = "/exports/lkeb-hpc/syousefi/Data/ASL2PET_high_res/AMUC_high_res/"
    data_path_LUMC = "/exports/lkeb-hpc/syousefi/Data/ASL2PET_high_res/LUMC_high_res/"
    
    _rd = _read_data(data_path_AMUC,data_path_LUMC)

    train_data, validation_data, test_data = _rd.read_data_path(0)

    if which_data == 1:
        data = validation_data
    elif which_data == 2:
        data = test_data
    elif which_data == 3:
        data = train_data

    asl_plchld = tf.placeholder(tf.float32, shape=[None, asl_size, asl_size, 1])
    t1_plchld = tf.placeholder(tf.float32, shape=[None, asl_size, asl_size, 1])
    pet_plchld = tf.placeholder(tf.float32, shape=[None, pet_size, pet_size, 1])
    asl_out_plchld = tf.placeholder(tf.float32, shape=[None, pet_size, pet_size, 1])
    hybrid_training_flag = tf.placeholder(tf.bool, name='hybrid_training_flag')
    ave_loss_vali = tf.placeholder(tf.float32)

    is_training = tf.placeholder(tf.bool, name='is_training')
    is_training_bn = tf.placeholder(tf.bool, name='is_training_bn')

    msdensnet = multi_stage_densenet()
    asl_y, pet_y = msdensnet.multi_stage_densenet(asl_img=asl_plchld,
                                                  t1_img=t1_plchld,
                                                  pet_img=pet_plchld,
                                                  hybrid_training_flag=hybrid_training_flag,
                                                  input_dim=asl_size,
                                                  is_training=is_training,
                                                  config=config,
                                                  )
    alpha = .84
    with tf.name_scope('cost'):
        ssim_asl = tf.reduce_mean(1 - SSIM(x1=asl_out_plchld, x2=asl_y, max_val=34.0)[0])
        loss_asl = alpha * ssim_asl + (1 - alpha) * tf.reduce_mean(huber(labels=asl_out_plchld, logit=asl_y))

        ssim_pet = tf.reduce_mean(1 - SSIM(x1=pet_plchld, x2=pet_y, max_val=2.1)[0])
        loss_pet = alpha * ssim_pet + (1 - alpha) * tf.reduce_mean(huber(labels=pet_plchld, logit=pet_y))

        cost_withpet = tf.reduce_mean(loss_asl + loss_pet)

        cost_withoutpet = loss_asl


    sess = tf.Session()
    saver = tf.train.Saver()
    parent_path = '/exports/lkeb-hpc/syousefi/Code/'
    chckpnt_dir = parent_path + Log + 'unet_checkpoints/'
    ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    _meas = _measure()
    copyfile('./test_semisupervised_multitast_hr_not1.py',
             parent_path + Log + out_dir + 'test_semisupervised_multitast_hr_not1.py')

    _image_class = image_class(data,
                               bunch_of_images_no=1,
                               is_training=0,
                               inp_size=asl_size,
                               out_size=pet_size)
    list_ssim = []
    list_name = []

    list_ssim_NC=[]
    list_ssim_HC=[]
    for scan in range(len(data)):
        ss = str(data[scan]['asl']).split("/")
        imm = _image_class.read_image(data[scan])
        try:
            os.mkdir(parent_path + Log + out_dir + ss[-3])
            os.mkdir(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[1])
        except:
            a = 1

        for img_indx in range(np.shape(imm[3])[0]):
            print('img_indx:%s' % (img_indx))

            name = ss[-3] + '_' + ss[-2] + '_' + str(img_indx)
            # name = (ss[10] + '_' + ss[11] + '_' + ss[12].split('%')[0]).split('_CT')[0]
            tic = time.time()
            t1 = imm[3][img_indx,
                 int(imm[3].shape[-1]/2) - int(asl_size / 2) - 1:int(imm[3].shape[-1]/2)  + int(asl_size / 2),
                 int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1]/2)  + int(asl_size / 2)]
            t1 = t1[np.newaxis, ..., np.newaxis]
            asl = imm[4][np.newaxis, img_indx,
                  int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(asl_size / 2),
                  int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1] / 2)+ int(asl_size / 2), np.newaxis]
            pet = imm[5][np.newaxis, img_indx,
                  int(imm[3].shape[-1] / 2)- int(pet_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(pet_size / 2),
                  int(imm[3].shape[-1] / 2)- int(pet_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(pet_size / 2), np.newaxis]

            [loss,pet_out,asl_out] = sess.run([ssim_pet,pet_y,asl_y],
                                   feed_dict={asl_plchld: asl,
                                              t1_plchld: t1,
                                              pet_plchld: pet,
                                              asl_out_plchld: asl[:,
                                                 int(asl_size / 2) - int(pet_size / 2) - 1:
                                                 int(asl_size / 2) + int(pet_size / 2),
                                                 int(asl_size / 2) - int(pet_size / 2) - 1:
                                                 int(asl_size / 2) + int(pet_size / 2), :],
                                              is_training: False,
                                              ave_loss_vali: -1,
                                              is_training_bn: False,
                                              hybrid_training_flag: False
                                              })

            ssim = 1 - loss
            list_ssim.append(ssim)
            str_nm = (ss[-3] + '_' + ss[-1].split(".")[0].split("ASL_")[1] + '_t1_' + name)
            if 'HN' in str_nm:
                list_ssim_NC.append(ssim)
            elif 'HY' in str_nm:
                list_ssim_HC.append(ssim)

            list_name.append(ss[-3] + '_' + ss[-1].split(".")[0].split("ASL_")[1] + '_t1_' + name)
            print(list_name[img_indx],': ',ssim)
            matplotlib.image.imsave(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[
                1] + '/t1_' + name + '.png', np.squeeze(t1), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[
                1] + '/asl_' + name + '_' + '.png', np.squeeze(asl), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[
                1] + '/pet_' + name + '_' + '.png', np.squeeze(pet), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[
                1] + '/res_asl' + name + '_' + str(ssim) + '.png', np.squeeze(asl_out), cmap='gray')
            matplotlib.image.imsave(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[
                1] + '/res_pet' + name + '_' + str(ssim) + '.png', np.squeeze(pet_out), cmap='gray')

            elapsed = time.time() - tic

    df = pd.DataFrame(list_ssim,
                      columns=pd.Index(['ssim'],
                                       name='Genus')).round(2)
    a = {'HC': list_ssim_HC, 'NC': list_ssim_NC}
    df2 = pd.DataFrame.from_dict(a, orient='index')
    df2.transpose()

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(parent_path + Log + out_dir + '/all_ssim.xlsx',
                            engine='xlsxwriter')
    writer2 = pd.ExcelWriter(parent_path + Log + out_dir + '/all_ssim.xlsx',
                             engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer2, sheet_name='Sheet2')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer2.save()

    print(parent_path + Log + out_dir + '/all_ssim.xlsx')


if __name__ == "__main__":
    Log = "Log_asl_pet/denseunet_hybrid_hr_not1_01/"
    which_data = 1
    if which_data == 1:
        out_dir = "0_vali_result/"
    elif which_data == 2:
        out_dir = "0_test_result/"
    elif which_data == 3:
        out_dir = "0_train_result/"

    test_all_nets(out_dir, Log, which_data=which_data)