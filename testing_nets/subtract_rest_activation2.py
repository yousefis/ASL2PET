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
from scipy import signal

mt_path= '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/0_test_result/'
#/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_hr_not1_02/0_test_result/
# path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_hr_not1_01/1_test_result/'
# path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_val_att2/residual_attention2_fold_1/0_test_result/'
st_path= '/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_plust1/single_plust1_fold_1/0_test_result/'
sub='CASL030'

tags=['rest','motor','checkerboard','tomenjerry']
which_tag1=1
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
indx=35
res_st_rest_pth=res_st_rest[indx]
res_mt_rest_pth=res_mt_rest[indx]
res_st_act_pth=res_st_act[indx]
res_mt_act_pth=res_mt_act[indx]
targ_rest_pth=targ_rest[indx]
targ_act_pth=targ_act[indx]


res_st_rest_I = sitk.ReadImage(res_st_rest_pth)
res_st_rest_img = sitk.GetArrayFromImage(res_st_rest_I)
res_mt_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_rest_pth))

sitk.WriteImage(sitk.ReadImage(res_st_rest_pth),'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_ST_rest_'+str(indx)+'.nii')

sitk.WriteImage(sitk.ReadImage(res_mt_rest_pth),'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_MT_rest_'+str(indx)+'.nii')



targ_rest_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_rest_pth))

sitk.WriteImage(sitk.ReadImage(targ_rest_pth),'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/ASL_rest'+str(indx)+'.nii')

res_st_act_img = sitk.GetArrayFromImage(sitk.ReadImage(res_st_act_pth))
sitk.WriteImage(sitk.ReadImage(res_st_act_pth),'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_ST_acti_'+str(indx)+'.nii')
res_mt_act_img = sitk.GetArrayFromImage(sitk.ReadImage(res_mt_act_pth))
sitk.WriteImage(sitk.ReadImage(res_mt_act_pth),'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_MT_acti_'+str(indx)+'.nii')

targ_act_img = sitk.GetArrayFromImage(sitk.ReadImage(targ_act_pth))
sitk.WriteImage(sitk.ReadImage(targ_act_pth),'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/ASL_acti_'+str(indx)+'.nii')



asl_neg_diff=-(targ_act_img - targ_rest_img)
II=sitk.GetImageFromArray(asl_neg_diff)
II.SetOrigin(res_st_rest_I.GetOrigin())
II.SetSpacing(res_st_rest_I.GetSpacing())
II.SetDirection(res_st_rest_I.GetDirection())
sitk.WriteImage(II,'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/ASL_rest_acti_'+str(indx)+'.nii')


asl_pos_diff=(targ_act_img - targ_rest_img)

II=sitk.GetImageFromArray(asl_pos_diff)
II.SetOrigin(res_st_rest_I.GetOrigin())
II.SetSpacing(res_st_rest_I.GetSpacing())
II.SetDirection(res_st_rest_I.GetDirection())
sitk.WriteImage(II,'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/ASL_acti_rest_'+str(indx)+'.nii',)

st_pet_neg_diff= res_st_rest_img - res_st_act_img
II=sitk.GetImageFromArray(st_pet_neg_diff)
II.SetOrigin(res_st_rest_I.GetOrigin())
II.SetSpacing(res_st_rest_I.GetSpacing())
II.SetDirection(res_st_rest_I.GetDirection())
sitk.WriteImage(II,'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_ST_rest_acti_'+str(indx)+'.nii')


st_pet_pos_diff=(res_st_act_img - res_st_rest_img)
II=sitk.GetImageFromArray(st_pet_pos_diff)
II.SetOrigin(res_st_rest_I.GetOrigin())
II.SetSpacing(res_st_rest_I.GetSpacing())
II.SetDirection(res_st_rest_I.GetDirection())
sitk.WriteImage(II,'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_ST_acti_rest_'+str(indx)+'.nii')



mt_pet_neg_diff= res_mt_rest_img - res_mt_act_img
II=sitk.GetImageFromArray(mt_pet_neg_diff)
II.SetOrigin(res_st_rest_I.GetOrigin())
II.SetSpacing(res_st_rest_I.GetSpacing())
II.SetDirection(res_st_rest_I.GetDirection())
sitk.WriteImage(II,'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_MT_rest_acti_'+str(indx)+'.nii')

mt_pet_pos_diff=(res_mt_act_img - res_mt_rest_img)
II=sitk.GetImageFromArray(mt_pet_pos_diff)
II.SetOrigin(res_st_rest_I.GetOrigin())
II.SetSpacing(res_st_rest_I.GetSpacing())
II.SetDirection(res_st_rest_I.GetDirection())
sitk.WriteImage(II,'/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/results4Thijs2/'+str(indx)+'/PET_MT_acti_rest_'+str(indx)+'.nii')



# cor = signal.correlate2d (mt_pet_pos_diff, asl_pos_diff)


dif_trg_size=np.shape(asl_neg_diff)[0]


asl_neg_diff= asl_neg_diff[int(dif_trg_size / 2) -
                           int(np.shape(st_pet_neg_diff)[0] / 2) - 1:
                int(dif_trg_size/2)+
                int(np.shape(st_pet_neg_diff)[0] / 2),
              int(dif_trg_size/2) -
              int(np.shape(st_pet_neg_diff)[0] / 2) -
              1:int(dif_trg_size/2)+
          int(np.shape(st_pet_neg_diff)[0] / 2)]

asl_pos_diff= asl_pos_diff[int(dif_trg_size / 2) -
                           int(np.shape(st_pet_neg_diff)[0] / 2) - 1:
                int(dif_trg_size/2)+
                int(np.shape(st_pet_neg_diff)[0] / 2),
              int(dif_trg_size/2) -
              int(np.shape(st_pet_neg_diff)[0] / 2) -
              1:int(dif_trg_size/2)+
          int(np.shape(st_pet_neg_diff)[0] / 2)]

mt_pet_neg_diff= mt_pet_neg_diff[int(dif_trg_size / 2) -
                                 int(np.shape(st_pet_neg_diff)[0] / 2) - 1:
                int(dif_trg_size/2)+
                int(np.shape(st_pet_neg_diff)[0] / 2),
          int(dif_trg_size/2) -
          int(np.shape(st_pet_neg_diff)[0] / 2) -
          1:int(dif_trg_size/2)+
          int(np.shape(st_pet_neg_diff)[0] / 2)]

mt_pet_pos_diff= mt_pet_pos_diff[int(dif_trg_size / 2) -
                                 int(np.shape(st_pet_neg_diff)[0] / 2) - 1:
                int(dif_trg_size/2)+
                int(np.shape(st_pet_neg_diff)[0] / 2),
            int(dif_trg_size/2) -
            int(np.shape(st_pet_neg_diff)[0] / 2) -
            1:int(dif_trg_size/2)+
          int(np.shape(st_pet_neg_diff)[0] / 2)]

targ_rest_img = targ_rest_img[int(dif_trg_size / 2) -
                              int(np.shape(st_pet_neg_diff)[0] / 2) - 1:
                   int(dif_trg_size/2)+
                   int(np.shape(st_pet_neg_diff)[0] / 2),
                int(dif_trg_size/2) -
                int(np.shape(st_pet_neg_diff)[0] / 2) -
                1:int(dif_trg_size/2)+
            int(np.shape(st_pet_neg_diff)[0] / 2)]

targ_act_img = targ_act_img[int(dif_trg_size / 2) -
                            int(np.shape(st_pet_neg_diff)[0] / 2) - 1:
                 int(dif_trg_size/2)+
                 int(np.shape(st_pet_neg_diff)[0] / 2),
               int(dif_trg_size/2) -
               int(np.shape(st_pet_neg_diff)[0] / 2) -
               1:int(dif_trg_size/2)+
           int(np.shape(st_pet_neg_diff)[0] / 2)]

# img1=tf.placeholder(tf.float32,shape=[1,None,None,1])
# img2=tf.placeholder(tf.float32,shape=[1,None,None,1])
# ssim = SSIM(img1,img2,max_val=max([np.max(dif_trg),np.max(dif_res)]))
# sess= tf.Session()
# sSiM_diff= sess.run(ssim,feed_dict={img1:np.expand_dims(np.expand_dims(dif_trg,0),-1),
#                                     img2:np.expand_dims(np.expand_dims(dif_res,0),-1)})


vmin_asl= min([np.min(targ_act_img),
           np.min(targ_rest_img),
           ])
vmax_asl= max([np.max(targ_act_img),
           np.max(targ_rest_img)])/1.5

plt.subplot(3,4, 1)
plt.imshow(targ_rest_img, vmin=vmin_asl, vmax=vmax_asl , cmap='jet')
plt.title('target/ASL rest')
plt.axis('off')
plt.colorbar()

plt.subplot(3,4, 2)
plt.imshow(targ_act_img, vmin=vmin_asl, vmax=vmax_asl , cmap='jet')
plt.title('target/ASL activation')
plt.axis('off')
plt.colorbar()


vmax_asl_diff= max([np.max(asl_neg_diff),
           ])/10
vmin_asl_diff= min([np.min(asl_neg_diff),
           ])/5


plt.subplot(3,4, 3)
plt.imshow(asl_neg_diff, vmin=vmin_asl_diff, vmax=vmax_asl_diff, cmap='jet')
plt.title('target/-ASL subtraction')
plt.axis('off')
plt.colorbar()

vmax_asl_diff= max([
           np.max(asl_pos_diff)])/5
vmin_asl_diff= min([
           np.min(asl_pos_diff),
           ])/10

plt.subplot(3,4, 4)
plt.imshow(asl_pos_diff, vmin=vmin_asl_diff, vmax=vmax_asl_diff, cmap='jet')
plt.title('target/+ASL subtraction')
plt.axis('off')
plt.colorbar()



vmax_pet= max([np.max(res_st_rest_img),
           np.max(res_st_act_img),
           np.max(res_mt_rest_img),
           np.max(res_mt_act_img)])
vmin_pet= max([np.min(res_st_rest_img),
           np.min(res_st_act_img),
           np.min(res_mt_rest_img),
           np.min(res_mt_act_img)])

plt.subplot(3,4, 5)
plt.imshow(res_st_rest_img, vmin=vmin_pet, vmax=vmax_pet, cmap='jet')
plt.title('result/ST/PET rest')
plt.axis('off')
plt.colorbar()

plt.subplot(3,4, 6)
plt.imshow(res_st_act_img, vmin=vmin_pet, vmax=vmax_pet, cmap='jet')
plt.title('result/ST/PET activation')
plt.axis('off')
plt.colorbar()

plt.subplot(3,4, 9)
plt.imshow(res_mt_rest_img, vmin=vmin_pet, vmax=vmax_pet, cmap='jet')
plt.title('result/MT/PET rest')
plt.axis('off')
plt.colorbar()

plt.subplot(3,4, 10)
plt.imshow(res_mt_act_img, vmin=vmin_pet, vmax=vmax_pet, cmap='jet')
plt.title('result/MT/PET activation')
plt.axis('off')
plt.colorbar()


vmin_diff =max([np.min(mt_pet_neg_diff),
            np.min(st_pet_neg_diff),
           ])/1.5
vmax_diff =max([
           np.max(mt_pet_neg_diff),
           np.max(st_pet_neg_diff)])/10

plt.subplot(3,4, 7)
plt.imshow(st_pet_neg_diff, vmin=vmin_diff, vmax=vmax_diff, cmap='jet')
plt.title('result/ST/-PET subtraction')
plt.axis('off')
plt.colorbar()

plt.subplot(3,4, 11)
plt.imshow(mt_pet_neg_diff, vmin=vmin_diff, vmax=vmax_diff, cmap='jet')
plt.title('result/MT/-PET subtraction')
plt.axis('off')
plt.colorbar()

vmax_diff =max([
           np.max(mt_pet_pos_diff),
           np.max(st_pet_pos_diff)])/1.5
vmin_diff =max([np.min(mt_pet_pos_diff),
            np.min(st_pet_pos_diff),
           ])/10

plt.subplot(3,4, 8)
plt.imshow(st_pet_pos_diff, vmin=vmin_diff, vmax=vmax_diff, cmap='jet')
plt.title('result/ST/+PET subtraction')
plt.axis('off')
plt.colorbar()



plt.subplot(3,4,12)
plt.imshow(mt_pet_pos_diff, vmin=vmin_diff, vmax=vmax_diff, cmap='jet')
plt.title('result/MT/+PET subtraction')
plt.axis('off')
plt.colorbar()


plt.show()




print(2)