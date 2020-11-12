import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, join
from os import listdir
from PIL import Image
import matplotlib
import os
import shutil
import SimpleITK as sitk
from functions.losses.ssim_loss import SSIM
from random import randrange
import pandas as pd
tags=['rest','motor','checkerboard','tomenjerry']
test_vali='0_test_result/'
def read_img_pth(which_tag,ext,pth):
    return  [join(pth +test_vali + sub + '/'+ tags[which_tag], f)
     for f in listdir(pth +test_vali+ sub + '/' + tags[which_tag])
     if ((isfile(join(pth +test_vali + sub + '/' + tags[which_tag], f))) and f.startswith(ext))]

if __name__=='__main__':
    # mt_path= '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/0_test_result/'
    # st_path= '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_plust1/single_plust1_fold_1/0_test_result/'

    all_paths= [#'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_not1_2/single_not1_fold_1/',
                '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_plust1/single_plust1_fold_1/',
                # '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_plust1_skipatt/single_plust1_skipatt_fold_1/',
                # '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/multitask_not1/denseunet_hybrid_not1_01/',
                # '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/plust1_noskippatt_noresidualatt/residual_attention2_plust1_no_resid_noskip_fold_1/',
                # '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_val_att2/residual_attention2_fold_1/',
                '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/',
                # '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/resi_skip_att_not1/residual_skip_not1_fold_1/'
                ]

    net_real_tags=[#'1_st_not1_noRA_noDA',
              'st_t1_noRA_noDA',
              # '3_st_not1_noRA_DA',
              # '4_mt_not1_noRA_noDA',
              # '5_mt_t1_noRA_noDA',
              # '6_mt_t1_RA_noDA',
              'mt_t1_RA_DA',
              # '8_mt_not1_RA_DA',
              ]
    net_tags=['a','b']

    out_path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/3D_volume/'


    subjects=['CASL030','CASL031','CASL032','CASL034']

    I = sitk.ReadImage('/exports/lkeb-hpc/syousefi/Data/asl_pet/pp01/PET/PET_W1_HN2.nii')
    origin = I.GetOrigin()
    spacing = I.GetSpacing()
    direction = I.GetDirection()

    hash_table_sub=[]
    hash_table_tag=[]
    hash_table_net=[]
    hash_table_type=[]


    # subjects=['CASL034']
    for sub in subjects:
        print(sub)
        os.mkdir(out_path + sub)
        for which_tag1 in [1, 2, 3]:
            print(tags[which_tag1])
            rnd_no1=randrange(2)
            rnd_no2=int(not rnd_no1)
            rnd_nos=[rnd_no1,rnd_no2]
            for r,t,nt in zip(rnd_nos,net_real_tags,net_tags) :
                p=all_paths[r]
                print(nt,net_real_tags[r],p)
                # item=[sub,tags[which_tag1],nt,net_real_tags[r]]
                hash_table_sub.append(sub)
                hash_table_type.append(tags[which_tag1])
                hash_table_tag.append(nt)
                hash_table_net.append(net_real_tags[r])

                res_mt_act= read_img_pth(which_tag=which_tag1,ext='res_pet',pth=p)
                res_mt_act.sort()
                which_tag2=0
                res_mt_rest= read_img_pth(which_tag=which_tag2,ext='res_pet',pth=p)
                res_mt_rest.sort()

                # for r,a,rt,at in zip(rest_imgs,activation_imgs,rest_target,activtion_target):
                x=219
                slices=108
                targ_act_pth_all=np.zeros([slices,x,x])
                res_st_act_img_all=np.zeros([slices,x,x])
                targ_rest_img_all=np.zeros([slices,x,x])
                res_mt_rest_img_all=np.zeros([slices,x,x])
                res_st_rest_img_all=np.zeros([slices,x,x])
                res_st_rest_I_all=np.zeros([slices,x,x])
                asl_neg_diff_all=np.zeros([slices,x,x])
                asl_pos_diff_all=np.zeros([slices,x,x])
                st_pet_neg_diff_all=np.zeros([slices,x,x])
                st_pet_pos_diff_all=np.zeros([slices,x,x])
                mt_pet_pos_diff_all=np.zeros([slices,x,x])
                mt_pet_neg_diff_all=np.zeros([slices,x,x])
                res_mt_act_img_all=np.zeros([slices,x,x])

                for indx in range(0,slices):
                    try:
                        res_mt_rest_pth=p+test_vali+ sub + '/' + tags[which_tag2]+'/res_pet'+(sub)+'_ASL_'+str(indx)+'_nan.nii'#res_mt_rest[indx]
                        res_mt_act_pth = p +test_vali+ sub + '/' + tags[which_tag1] + '/res_pet' + (sub) + '_ASL_' + str(indx) + '_nan.nii'  # res_mt_act[indx]
                        res_mt_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_rest_pth))
                        res_mt_act_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_act_pth))

                        targ_rest_pth = p+test_vali + sub + '/' + tags[which_tag2] + '/asl_' + (sub) + '_ASL_' + str(
                            indx) + '_.nii'  # targ_rest[indx]
                        targ_act_pth = p+test_vali + sub + '/' + tags[which_tag1] + '/asl_' + (sub) + '_ASL_' + str(
                            indx) + '_.nii'  # targ_act[indx]
                        targ_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_rest_pth))
                        targ_act_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_act_pth))
                    except:
                        res_mt_rest_pth = p + test_vali + sub + '/' + tags[which_tag2] + '/res_pet' + (
                            sub) + '_ASL_' + str(indx) + '_nan.mha'  # res_mt_rest[indx]
                        res_mt_act_pth = p + test_vali + sub + '/' + tags[which_tag1] + '/res_pet' + (
                            sub) + '_ASL_' + str(indx) + '_nan.mha'  # res_mt_act[indx]
                        res_mt_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_rest_pth))
                        res_mt_act_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_act_pth))

                        targ_rest_pth = p + test_vali + sub + '/' + tags[which_tag2] + '/asl_' + (sub) + '_ASL_' + str(
                            indx) + '_.mha'  # targ_rest[indx]
                        targ_act_pth = p + test_vali + sub + '/' + tags[which_tag1] + '/asl_' + (sub) + '_ASL_' + str(
                            indx) + '_.mha'  # targ_act[indx]
                        targ_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_rest_pth))
                        targ_act_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_act_pth))

                    res_mt_rest_img_all[indx, :, :] = res_mt_rest_img
                    res_mt_act_img_all[indx, :, :] = res_mt_act_img
                    mt_pet_neg_diff_all[indx, :, :] = res_mt_rest_img - res_mt_act_img
                    mt_pet_pos_diff_all[indx, :, :] = (res_mt_act_img - res_mt_rest_img)


                    targ_rest_img_all[indx, :, :] = targ_rest_img[6:-6, 6:-6]
                    targ_act_pth_all[indx, :, :] = targ_act_img[6:-6, 6:-6]
                    asl_neg_diff_all[indx, :, :] = -(targ_act_img - targ_rest_img)[6:-6, 6:-6]
                    asl_pos_diff_all[indx, :, :] = (targ_act_img - targ_rest_img)[6:-6, 6:-6]

                FIF= sitk.FlipImageFilter()
                FIF.SetFlipAxes([False,True,False])

                #===================================
                try:
                    os.mkdir(out_path + sub + '/' + tags[which_tag1])
                    II_targ_rest_img_all = FIF.Execute(sitk.GetImageFromArray(targ_rest_img_all))
                    II_targ_rest_img_all.SetOrigin(origin)
                    II_targ_rest_img_all.SetSpacing(spacing)
                    II_targ_rest_img_all.SetDirection(direction)
                    sitk.WriteImage(II_targ_rest_img_all,
                                    out_path + sub + '/' + '/' + tags[which_tag1] + '/ASL_rest' + '.nii')

                    II_targ_act_pth_all = FIF.Execute(sitk.GetImageFromArray(targ_act_pth_all))
                    II_targ_act_pth_all.SetOrigin(origin)
                    II_targ_act_pth_all.SetSpacing(spacing)
                    II_targ_act_pth_all.SetDirection(direction)
                    sitk.WriteImage(II_targ_act_pth_all,
                                    out_path + sub + '/' + '/' + tags[which_tag1] + '/ASL_acti' + '.nii')

                    II_asl_neg_diff = FIF.Execute(sitk.GetImageFromArray(asl_neg_diff_all))
                    II_asl_neg_diff.SetOrigin(origin)
                    II_asl_neg_diff.SetSpacing(spacing)
                    II_asl_neg_diff.SetDirection(direction)
                    sitk.WriteImage(II_asl_neg_diff,
                                    out_path + sub + '/' + tags[which_tag1] + '/ASL_rest_acti' + '.nii')

                    II_asl_pos_diff = FIF.Execute(sitk.GetImageFromArray(asl_pos_diff_all))
                    II_asl_pos_diff.SetOrigin(origin)
                    II_asl_pos_diff.SetSpacing(spacing)
                    II_asl_pos_diff.SetDirection(direction)
                    sitk.WriteImage(II_asl_pos_diff,
                                    out_path + sub + '/' + tags[which_tag1] + '/ASL_acti_rest'+ '.nii', )

                except:
                    a=1

                II_res_mt_rest_img_all = FIF.Execute(sitk.GetImageFromArray(res_mt_rest_img_all))
                II_res_mt_rest_img_all.SetOrigin(origin)
                II_res_mt_rest_img_all.SetSpacing(spacing)
                II_res_mt_rest_img_all.SetDirection(direction)
                sitk.WriteImage(II_res_mt_rest_img_all,
                                out_path + sub+'/'+tags[which_tag1]+ '/PET_rest_' +nt+ '.nii')

                II_res_mt_act_img_all = FIF.Execute(sitk.GetImageFromArray(res_mt_act_img_all))
                II_res_mt_act_img_all.SetOrigin(origin)
                II_res_mt_act_img_all.SetSpacing(spacing)
                II_res_mt_act_img_all.SetDirection(direction)
                sitk.WriteImage(II_res_mt_act_img_all,
                                out_path + sub+'/'+tags[which_tag1]+ '/PET_acti_' +nt+ '.nii')

                II_mt_pet_neg_diff=FIF.Execute(sitk.GetImageFromArray(mt_pet_neg_diff_all))
                II_mt_pet_neg_diff.SetOrigin(origin)
                II_mt_pet_neg_diff.SetSpacing(spacing)
                II_mt_pet_neg_diff.SetDirection(direction)
                sitk.WriteImage(II_mt_pet_neg_diff,out_path+ sub+'/'+tags[which_tag1]+'/PET_rest_acti_'+nt+'.nii')

                II_mt_pet_pos_diff=FIF.Execute(sitk.GetImageFromArray(mt_pet_pos_diff_all))
                II_mt_pet_pos_diff.SetOrigin(origin)
                II_mt_pet_pos_diff.SetSpacing(spacing)
                II_mt_pet_pos_diff.SetDirection(direction)
                sitk.WriteImage(II_mt_pet_pos_diff,out_path+ sub+'/'+tags[which_tag1]+'/PET_acti_rest_'+nt+'.nii')

    df = pd.DataFrame.from_dict({'Subject': hash_table_sub,
                                 'Type': hash_table_type,
                                 'Tag': hash_table_tag,
                                 'CNN': hash_table_net,
                                 })
    writer = pd.ExcelWriter(out_path + '/tags.xlsx',
                            engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()