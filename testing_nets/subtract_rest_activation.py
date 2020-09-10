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
path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/0_test_result/'
#/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_hr_not1_02/0_test_result/
# path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_hr_not1_01/1_test_result/'
# path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_val_att2/residual_attention2_fold_1/0_test_result/'
path='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/01_cross_validation/single_plust1/single_plust1_fold_1/0_test_result/'
sub='CASL030'

tags=['rest','motor','checkerboard','tomenjerry']
which_tag1=1
activation_imgs= [join(path+ sub + '/' +tags[which_tag1], f)
                  for f in listdir(path + sub + '/' + tags[which_tag1])
                  if ((isfile(join(path+ sub + '/' + tags[which_tag1], f))) and f.startswith('res_pet'))]
activtion_target= [join(path+ sub + '/' +tags[which_tag1], f)
                   for f in listdir(path + sub + '/' + tags[which_tag1])
                   if ((isfile(join(path+ sub + '/' + tags[which_tag1], f))) and f.startswith('asl_'))]
activation_imgs.sort()
activtion_target.sort()
which_tag2=0
rest_imgs= [join(path+ sub + '/' +tags[which_tag2], f)
            for f in listdir(path + sub + '/' + tags[which_tag2])
            if ((isfile(join(path+ sub + '/' + tags[which_tag2], f))) and f.startswith('res_pet'))]

rest_target= [join(path+ sub + '/' +tags[which_tag2], f)
              for f in listdir(path + sub + '/' + tags[which_tag2])
              if ((isfile(join(path+ sub + '/' + tags[which_tag2], f))) and f.startswith('asl_'))]


rest_imgs.sort()
rest_target.sort()
# for r,a,rt,at in zip(rest_imgs,activation_imgs,rest_target,activtion_target):
indx=30
r=rest_imgs[indx]
a=activation_imgs[indx]
rt=rest_target[indx]
at=activtion_target[indx]
rest = sitk.GetArrayFromImage(sitk.ReadImage(r))

rest_trg = sitk.GetArrayFromImage(sitk.ReadImage(rt))

act = sitk.GetArrayFromImage(sitk.ReadImage(a))

act_trg = sitk.GetArrayFromImage(sitk.ReadImage(at))


vmin=min([np.min(rest),np.min(rest_trg),np.min(act),np.min(act)])/10
vmax=max([np.max(rest),np.max(rest_trg),np.max(act),np.max(act)])/10
dif_trg=-(act_trg-rest_trg)
dif_res=-(act-rest)
dif_trg_size=np.shape(dif_trg)[0]


dif_trg=dif_trg[int(dif_trg_size/2)-
                int(np.shape(dif_res)[0]/2)-1:
                int(dif_trg_size/2)+
                int(np.shape(dif_res)[0]/2),
        int(dif_trg_size/2)-
        int(np.shape(dif_res)[0]/2)-
        1:int(dif_trg_size/2)+
          int(np.shape(dif_res)[0]/2)]

rest_trg =rest_trg[int(dif_trg_size/2)-
                   int(np.shape(dif_res)[0]/2)-1:
                   int(dif_trg_size/2)+
                   int(np.shape(dif_res)[0]/2),
          int(dif_trg_size/2)-
          int(np.shape(dif_res)[0]/2)-
          1:int(dif_trg_size/2)+
            int(np.shape(dif_res)[0]/2)]

act_trg =act_trg[int(dif_trg_size/2)-
                 int(np.shape(dif_res)[0]/2)-1:
                 int(dif_trg_size/2)+
                 int(np.shape(dif_res)[0]/2),
         int(dif_trg_size/2)-
         int(np.shape(dif_res)[0]/2)-
         1:int(dif_trg_size/2)+
           int(np.shape(dif_res)[0]/2)]

# img1=tf.placeholder(tf.float32,shape=[1,None,None,1])
# img2=tf.placeholder(tf.float32,shape=[1,None,None,1])
# ssim = SSIM(img1,img2,max_val=max([np.max(dif_trg),np.max(dif_res)]))
# sess= tf.Session()
# sSiM_diff= sess.run(ssim,feed_dict={img1:np.expand_dims(np.expand_dims(dif_trg,0),-1),
#                                     img2:np.expand_dims(np.expand_dims(dif_res,0),-1)})



plt.subplot(2,3, 1)
plt.imshow(rest_trg, vmin=vmin, vmax=np.max(rest_trg)/1.5, cmap='jet')
plt.title('target/ASL rest')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(act_trg, vmin=vmin, vmax=np.max(rest_trg)/1.5, cmap='jet')
plt.title('target/ASL activation')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dif_trg, vmin=vmin, vmax=np.max(dif_trg), cmap='jet')
plt.title('target/ASL subtraction')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(rest, vmin=vmin, vmax=np.max(act)*1.2, cmap='jet')
plt.title('result/PET rest')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(act, vmin=vmin, vmax=np.max(act)*1.2, cmap='jet')
plt.title('result/PET activation')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(dif_res, vmin=vmin, vmax=np.max(dif_res), cmap='jet')
plt.title('result/PET subtraction')
plt.axis('off')

# plt.subplot(2, 4, 8)
# plt.imshow(np.squeeze(sSiM_diff[-1]), vmin=vmin, vmax=vmax, cmap='jet')
# plt.title('result/ssimap')
# print(sSiM_diff)
# plt.colorbar()

plt.show()


print(2)