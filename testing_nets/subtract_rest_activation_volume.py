import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, join
from os import listdir
from PIL import Image
import matplotlib
import tensorflow as tf
import os
import SimpleITK as sitk
from functions.losses.ssim_loss import SSIM
mt_path= '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/0_test_result/'
#/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_hr_not1_02/0_test_result/
# path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_hr_not1_01/1_test_result/'
# path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_val_att2/residual_attention2_fold_1/0_test_result/'
st_path= '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_plust1/single_plust1_fold_1/0_test_result/'
subjects=['CASL030','CASL031','CASL032','CASL034']
# subjects=['CASL034']
for sub in subjects:
    tags=['rest','motor','checkerboard','tomenjerry']
    print(sub)

    for which_tag1 in [1,2,3]:
        print(tags[which_tag1])
        I=sitk.ReadImage('/exports/lkeb-hpc/syousefi/Data/asl_pet/pp01/PET/PET_W1_HN2.nii')
        origin=I.GetOrigin()
        spacing=I.GetSpacing()
        direction=I.GetDirection()


        res_mt_act= [join(mt_path + sub + '/' + tags[which_tag1], f)
                     for f in listdir(mt_path + sub + '/' + tags[which_tag1])
                     if ((isfile(join(mt_path + sub + '/' + tags[which_tag1], f))) and f.startswith('res_pet'))]


        res_st_act= [join(st_path + sub + '/' + tags[which_tag1], f)
                     for f in listdir(st_path + sub + '/' + tags[which_tag1])
                     if ((isfile(join(st_path + sub + '/' + tags[which_tag1], f))) and f.startswith('res_pet'))]
        targ_act= [join(st_path + sub + '/' + tags[which_tag1], f)
                   for f in listdir(st_path + sub + '/' + tags[which_tag1])
                   if ((isfile(join(st_path + sub + '/' + tags[which_tag1], f))) and f.startswith('asl_'))]
        res_st_act.sort()
        res_mt_act.sort()
        targ_act.sort()
        which_tag2=0
        res_st_rest= [join(st_path + sub + '/' + tags[which_tag2], f)
                      for f in listdir(st_path + sub + '/' + tags[which_tag2])
                      if ((isfile(join(st_path + sub + '/' + tags[which_tag2], f))) and f.startswith('res_pet'))]

        res_mt_rest= [join(mt_path + sub + '/' + tags[which_tag2], f)
                      for f in listdir(mt_path + sub + '/' + tags[which_tag2])
                      if ((isfile(join(mt_path + sub + '/' + tags[which_tag2], f))) and f.startswith('res_pet'))]


        targ_rest= [join(st_path + sub + '/' + tags[which_tag2], f)
                    for f in listdir(st_path + sub + '/' + tags[which_tag2])
                    if ((isfile(join(st_path + sub + '/' + tags[which_tag2], f))) and f.startswith('asl_'))]


        res_st_rest.sort()
        res_mt_rest.sort()
        targ_rest.sort()
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
            res_st_rest_pth=st_path+ sub + '/' + tags[which_tag2]+'/res_pet'+(sub)+'_ASL_'+str(indx)+'_nan.nii'#res_st_rest[indx]
            res_mt_rest_pth=mt_path+ sub + '/' + tags[which_tag2]+'/res_pet'+(sub)+'_ASL_'+str(indx)+'_nan.mha'#res_mt_rest[indx]
            res_st_act_pth=st_path+ sub + '/' + tags[which_tag1]+'/res_pet'+(sub)+'_ASL_'+str(indx)+'_nan.nii'#res_st_act[indx]
            res_mt_act_pth=mt_path+ sub + '/' + tags[which_tag1]+'/res_pet'+(sub)+'_ASL_'+str(indx)+'_nan.mha'#res_mt_act[indx]
            targ_rest_pth=st_path+ sub + '/' + tags[which_tag2]+'/asl_'+(sub)+'_ASL_'+str(indx)+'_.nii'#targ_rest[indx]
            targ_act_pth=st_path+ sub + '/' + tags[which_tag1]+'/asl_'+(sub)+'_ASL_'+str(indx)+'_.nii'#targ_act[indx]


            res_st_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(res_st_rest_pth))
            res_mt_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_rest_pth))
            targ_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_rest_pth))
            res_st_act_img = sitk.GetArrayFromImage(sitk.ReadImage(res_st_act_pth))
            targ_act_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_act_pth))
            res_mt_act_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_act_pth))

            targ_act_pth_all[indx,:,:]=targ_act_img[6:-6,6:-6]
            res_st_act_img_all[indx,:,:]=res_st_act_img
            targ_rest_img_all[indx,:,:]=targ_rest_img[6:-6,6:-6]
            res_mt_rest_img_all[indx,:,:]=res_mt_rest_img
            res_st_rest_img_all[indx,:,:]=res_st_rest_img
            asl_neg_diff_all[indx,:,:]=-(targ_act_img - targ_rest_img)[6:-6,6:-6]
            asl_pos_diff_all[indx,:,:]=(targ_act_img - targ_rest_img)[6:-6,6:-6]
            st_pet_neg_diff_all[indx,:,:]= res_st_rest_img - res_st_act_img
            st_pet_pos_diff_all[indx,:,:]=(res_st_act_img - res_st_rest_img)
            res_mt_act_img_all[indx,:,:] = res_mt_act_img
            mt_pet_neg_diff_all[indx,:,:]= res_mt_rest_img - res_mt_act_img
            mt_pet_pos_diff_all[indx,:,:]=(res_mt_act_img - res_mt_rest_img)



        out_path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+sub+'/'+tags[which_tag1]+'/'
        II_targ_act_pth_all=sitk.GetImageFromArray(targ_act_pth_all)
        II_targ_act_pth_all.SetOrigin(origin)
        II_targ_act_pth_all.SetSpacing(spacing)
        II_targ_act_pth_all.SetDirection(direction)
        sitk.WriteImage(II_targ_act_pth_all,
                        out_path + '/ASL_acti' + '.nii')


        II_res_st_rest_img_all=sitk.GetImageFromArray(res_st_rest_img_all)
        II_res_st_rest_img_all.SetOrigin(origin)
        II_res_st_rest_img_all.SetSpacing(spacing)
        II_res_st_rest_img_all.SetDirection(direction)
        sitk.WriteImage(II_res_st_rest_img_all,
                        out_path + '/PET_ST_rest' + '.nii')

        II_res_mt_rest_img_all=sitk.GetImageFromArray(res_mt_rest_img_all)
        II_res_mt_rest_img_all.SetOrigin(origin)
        II_res_mt_rest_img_all.SetSpacing(spacing)
        II_res_mt_rest_img_all.SetDirection(direction)
        sitk.WriteImage(II_res_mt_rest_img_all,
                        out_path + '/PET_MT_rest' + '.nii')

        II_targ_rest_img_all=sitk.GetImageFromArray(targ_rest_img_all)
        II_targ_rest_img_all.SetOrigin(origin)
        II_targ_rest_img_all.SetSpacing(spacing)
        II_targ_rest_img_all.SetDirection(direction)
        sitk.WriteImage(II_targ_rest_img_all,
                        out_path + '/ASL_rest' + '.nii')

        II_res_st_act_img_all=sitk.GetImageFromArray(res_st_act_img_all)
        II_res_st_act_img_all.SetOrigin(origin)
        II_res_st_act_img_all.SetSpacing(spacing)
        II_res_st_act_img_all.SetDirection(direction)
        sitk.WriteImage(II_res_st_act_img_all,
                        out_path + '/PET_ST_acti' + '.nii')

        II_res_mt_act_img_all=sitk.GetImageFromArray(res_mt_act_img_all)
        II_res_mt_act_img_all.SetOrigin(origin)
        II_res_mt_act_img_all.SetSpacing(spacing)
        II_res_mt_act_img_all.SetDirection(direction)
        sitk.WriteImage(II_res_mt_act_img_all,
                        out_path + '/PET_MT_acti' + '.nii')

        II_asl_neg_diff=sitk.GetImageFromArray(asl_neg_diff_all)
        II_asl_neg_diff.SetOrigin(origin)
        II_asl_neg_diff.SetSpacing(spacing)
        II_asl_neg_diff.SetDirection(direction)
        sitk.WriteImage(II_asl_neg_diff,out_path+'/ASL_rest_acti'+'.nii')

        II_asl_pos_diff=sitk.GetImageFromArray(asl_pos_diff_all)
        II_asl_pos_diff.SetOrigin(origin)
        II_asl_pos_diff.SetSpacing(spacing)
        II_asl_pos_diff.SetDirection(direction)

        sitk.WriteImage(II_asl_pos_diff,out_path+'/ASL_acti_rest'+'.nii',)
        II_st_pet_neg_diff=sitk.GetImageFromArray(st_pet_neg_diff_all)
        II_st_pet_neg_diff.SetOrigin(origin)
        II_st_pet_neg_diff.SetSpacing(spacing)
        II_st_pet_neg_diff.SetDirection(direction)
        sitk.WriteImage(II_st_pet_neg_diff,out_path+'/PET_ST_rest_acti'+'.nii')

        II_st_pet_pos_diff=sitk.GetImageFromArray(st_pet_pos_diff_all)
        II_st_pet_pos_diff.SetOrigin(origin)
        II_st_pet_pos_diff.SetSpacing(spacing)
        II_st_pet_pos_diff.SetDirection(direction)
        sitk.WriteImage(II_st_pet_pos_diff,out_path+'/PET_ST_acti_rest'+'.nii')

        II_mt_pet_neg_diff=sitk.GetImageFromArray(mt_pet_neg_diff_all)
        II_mt_pet_neg_diff.SetOrigin(origin)
        II_mt_pet_neg_diff.SetSpacing(spacing)
        II_mt_pet_neg_diff.SetDirection(direction)
        sitk.WriteImage(II_mt_pet_neg_diff,out_path+'/PET_MT_rest_acti'+'.nii')

        II_mt_pet_pos_diff=sitk.GetImageFromArray(mt_pet_pos_diff_all)
        II_mt_pet_pos_diff.SetOrigin(origin)
        II_mt_pet_pos_diff.SetSpacing(spacing)
        II_mt_pet_pos_diff.SetDirection(direction)
        sitk.WriteImage(II_mt_pet_pos_diff,out_path+'/PET_MT_acti_rest'+'.nii')

