# do test process
import matplotlib.pyplot as plt
import matplotlib

import time
from shutil import copyfile
from functions.losses.ssim_loss import SSIM,MSE, PSNR
# from functions.reader.data_reader_hybrid_hr import _read_data
from functions.reader.data_reader_hybrid_hr_cross_validate import _read_data
# from functions.cnn.hybrid_multi_task_cnn import multi_stage_densenet
from functions.cnn.hybrid_multi_task_cnn_hr_not1 import multi_stage_densenet
from functions.measures.measures import _measure
from functions.reader.image_class_hybrid_hr import image_class
import SimpleITK as sitk
import tensorflow as tf
import os
# import wandb

# from functions.networks.dense_unet2 import _densenet_unet
import logging
from functions.cnn.hybrid_multi_task_cnn_hr_not1 import multi_stage_densenet
# import wandb
# from functions.reader.data_reader_hybrid_hr import *
from functions.reader.image_class_hybrid_hr import *
from functions.losses.ssim_loss import  SSIM
import functions.threads.fill_thread_hr as fill_thread
import functions.threads.read_thread_hr as read_thread
import psutil
# calculate the dice coefficient
from shutil import copyfile
import functions.reader.patch_extractor_hybrid_hr as _patch_extractor_thread
from functions.reader.data_reader_hybrid_hr_cross_validate import _read_data
from functions.losses.L1 import huber
from functions.losses.ssim_loss import SSIM,MSE, PSNR


import pandas as pd
import numpy as np
import SimpleITK as sitk
from functions.losses.L1 import huber
# do test process
import matplotlib.pyplot as plt
import matplotlib

import time
from shutil import copyfile
from functions.losses.ssim_loss import SSIM
from functions.reader.data_reader_hybrid_hr import *
# from functions.reader.data_reader_hybrid_hr import _read_data
# from functions.reader.data_reader_hybrid_rest import _read_data
from functions.measures.measures import _measure
from functions.reader.image_class_hybrid_hr import image_class
import SimpleITK as sitk

# import wandb


import pandas as pd
import numpy as np
import SimpleITK as sitk
from functions.losses.L1 import huber
def print_ave_std(ssim_hc,ssim_nc,psnr_hc,psnr_nc,mse_hc,mse_nc):
    print('HC:')
    print(np.average(np.array(ssim_hc)).round(2), np.std(np.array(ssim_hc)).round(2))
    print(np.average(np.array(mse_hc)).round(2), np.std(np.array(mse_hc)).round(2))
    print(np.average(np.array(psnr_hc)).round(2), np.std(np.array(psnr_hc)).round(1))

    print('NC:')
    print(np.average(np.array(ssim_nc)).round(2), np.std(np.array(ssim_nc)).round(2))
    print(np.average(np.array(mse_nc)).round(2), np.std(np.array(mse_nc)).round(2))
    print(np.average(np.array(psnr_nc)).round(2), np.std(np.array(psnr_nc)).round(1))


    print('All:')
    print(np.average(ssim_hc+ssim_nc), np.std(ssim_hc+ssim_nc))
    print(np.average(psnr_hc + psnr_nc), np.std(psnr_hc + psnr_nc))
    print(np.average(mse_hc+mse_nc), np.std(mse_hc+mse_nc))


eps = 1E-5
plot_tag = 'densenet'
config = [3, 3, 5, 3, 3]

asl_size = 231
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


def test_all_nets(out_dir, Log, which_data,fold):
    data_path_AMUC = "/exports/lkeb-hpc/syousefi/Data/ASL2PET_high_res/AMUC_high_res/"
    data_path_LUMC = "/exports/lkeb-hpc/syousefi/Data/ASL2PET_high_res/LUMC_high_res/"

    _rd = _read_data(data_path_AMUC, data_path_LUMC)

    train_data, validation_data, test_data = _rd.read_data_path(fold=fold,average_no=0)

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
    mse = MSE(x1=pet_plchld, x2=pet_y)
    psnr = PSNR(x1=pet_plchld, x2=pet_y)

    sess = tf.Session()
    saver = tf.train.Saver()
    parent_path = '/exports/lkeb-hpc/syousefi/Code/'
    chckpnt_dir = parent_path + Log + 'unet_checkpoints/'
    ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, chckpnt_dir+'densenet_unet_inter_epoch0_point15000.ckpt-15000')
    _meas = _measure()
    copyfile('test_semisupervised_multitast_hr_rest_not1_noresdatt.py',
             parent_path + Log + out_dir + 'test_semisupervised_multitast_hr_rest_not1_noresdatt.py')

    _image_class = image_class(data,
                               bunch_of_images_no=1,
                               is_training=0,
                               inp_size=asl_size,
                               out_size=pet_size)
    list_ssim_pet = []
    list_ssim_asl = []
    list_name = []
    list_ssim_NC = []
    list_mse_NC = []
    list_psnr_NC = []
    list_ssim_HC = []
    list_mse_HC = []
    list_psnr_HC = []
    for scan in range(len(data)):
        ss = str(data[scan]['asl']).split("/")
        imm = _image_class.read_image(data[scan])
        try:
            # print(parent_path + Log + out_dir + ss[-3])
            os.mkdir(parent_path + Log + out_dir + ss[-3])

        except:
            a = 1
        # print(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[1])
        try:
            os.mkdir(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[1])
        except:
            try:
                os.mkdir(parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0])
            except:
                print(1)
        for img_indx in range(np.shape(imm[3])[0]):
            print('img_indx:%s' % (img_indx))

            name = ss[-3] + '_' + ss[-2] + '_' + str(img_indx)
            # name = (ss[10] + '_' + ss[11] + '_' + ss[12].split('%')[0]).split('_CT')[0]
            tic = time.time()
            t1 = imm[3][img_indx,
                 int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(asl_size / 2),
                 int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(asl_size / 2)]
            t1 = t1[np.newaxis, ..., np.newaxis]
            asl = imm[4][np.newaxis, img_indx,
                  int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(asl_size / 2),
                  int(imm[3].shape[-1] / 2) - int(asl_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(asl_size / 2),
                  np.newaxis]
            if np.size(imm[5]) > 1:
                pet = imm[5][np.newaxis, img_indx,
                      int(imm[3].shape[-1] / 2) - int(pet_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(pet_size / 2),
                      int(imm[3].shape[-1] / 2) - int(pet_size / 2) - 1:int(imm[3].shape[-1] / 2) + int(pet_size / 2),
                      np.newaxis]
                hybrid_training_f = True
            else:
                hybrid_training_f = False
                pet = np.reshape([None] * pet_size * pet_size, [pet_size, pet_size])
                pet = pet[..., np.newaxis]
                pet = pet[np.newaxis, ...]

            # if hybrid_training_f:
            [loss, psnr1, mse1, pet_out, asl_out] = sess.run([ssim_pet, psnr, mse, pet_y, asl_y],
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
            # if hybrid_training_f:
            ssim = 1 - loss
            list_ssim_pet.append(ssim)


            try:
                str_nm = (ss[-3] + '_' + ss[-1].split(".")[0].split("ASL_")[1] + '_t1_' + name)
            except:
                str_nm = (ss[-3] + '_' + ss[-1].split(".")[0] + '_t1_' + name)

            if 'HN' in str_nm:
                list_ssim_NC.append(ssim)
                list_psnr_NC.append(psnr1)
                list_mse_NC.append(mse1)
            elif 'HY' in str_nm:
                list_ssim_HC.append(ssim)
                list_psnr_HC.append(psnr1)
                list_mse_HC.append(mse1)

            try:
                list_name.append(ss[-3] + '_' + ss[-1].split(".")[0].split("ASL_")[1] + '_t1_' + name)
                nm_fig = parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0].split("ASL_")[1]
            except:
                list_name.append(ss[-3] + '_' + ss[-1].split(".")[0] + '_t1_' + name)
                nm_fig = parent_path + Log + out_dir + ss[-3] + '/' + ss[-1].split(".")[0]
            print(list_name[img_indx], ': ', ssim)

            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(t1)), nm_fig + '/t1_' + name + '.nii')
            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(asl)), nm_fig + '/asl_' + name + '_' + '.nii')
            if hybrid_training_f:
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(pet)), nm_fig + '/pet_' + name + '_' + '.nii')
            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(pet_out)),
                            nm_fig + '/res_pet' + name + '_' + str(ssim) + '.nii')
            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(asl_out)),
                            nm_fig + '/res_asl' + name + '_' + str(ssim) + '.nii')

            elapsed = time.time() - tic
    df = pd.DataFrame.from_dict({'SSIM_HC': list_ssim_HC,
                                 'MSE_HC': list_mse_HC,
                                 'PSNR_HC': list_psnr_HC, })
    df2 = pd.DataFrame.from_dict({'SSIM_NC': list_ssim_NC,
                                  'MSE_NC': list_mse_NC,
                                  'PSNR_NC': list_psnr_NC})
    writer = pd.ExcelWriter(parent_path + Log + out_dir + '/all_ssim.xlsx',
                            engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
    writer.save()
    # print(np.average(list_ssim_HC),np.average(list_mse_HC),np.average(list_psnr_HC))
    # print(np.std(list_ssim_HC),np.std(list_mse_HC),np.std(list_psnr_HC))
    #
    # print(np.average(list_ssim_NC), np.average(list_mse_NC), np.average(list_psnr_NC))
    # print(np.std(list_ssim_NC), np.std(list_mse_NC), np.std(list_psnr_NC))

    # print(np.average(list_ssim_NC), np.average(list_mse_NC), np.average(list_psnr_NC))
    # print(np.std(list_ssim_NC), np.std(list_mse_NC), np.std(list_psnr_NC))

    # df = pd.DataFrame(list_ssim_pet,
    #                   columns=pd.Index(['ssim'],
    #                                    name='Genus')).round(2)
    #
    # a = {'HC': list_ssim_HC, 'NC': list_ssim_NC}
    # df2 = pd.DataFrame.from_dict(a, orient='index')
    # df2.transpose()
    #
    # # Create a Pandas Excel writer using XlsxWriter as the engine.
    # writer = pd.ExcelWriter(parent_path + Log + out_dir + '/all_ssim.xlsx',
    #                         engine='xlsxwriter')
    # writer2 = pd.ExcelWriter(parent_path + Log + out_dir + '/all_ssim.xlsx',
    #                          engine='xlsxwriter')
    # # Convert the dataframe to an XlsxWriter Excel object.
    # df.to_excel(writer, sheet_name='Sheet1')
    # df2.to_excel(writer2, sheet_name='Sheet2')
    # # Close the Pandas Excel writer and output the Excel file.
    # writer.save()
    # writer2.save()
    print_ave_std(list_ssim_HC, list_ssim_NC, list_psnr_HC, list_psnr_NC, list_mse_HC, list_mse_NC)
    print(parent_path + Log + out_dir + '/all_ssim.xlsx')





if __name__ == "__main__":
    fold=1
    # Log = 'Log_asl_pet/rest/00_cross_validation/not1_noskippatt_noresidualatt/residual_attention2_not1_no_residualatt_fold_' + str(fold) + '/'
    Log = '/Log_asl_pet/rest/01_cross_validation/multitask_not1/denseunet_hybrid_not1_0' + str(fold) + '/'
    which_data =2
    if which_data == 1:
        out_dir = "0_vali_result/"
    elif which_data == 2:
        out_dir = "0_test_result/"
    elif which_data == 3:
        out_dir = "0_train_result/"

    test_all_nets(out_dir, Log, which_data=which_data,fold=fold)