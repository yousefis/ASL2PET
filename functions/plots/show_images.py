import matplotlib.pyplot as plt
import SimpleITK as sitk
import datetime
import os
import shutil
import numpy as np
def read_img(sub,path):
    nm = 'res_petpp03_ASL_' + str(inx)
    path_res = [(os.path.join(path + sub + f)) for f in os.listdir(os.path.join(path + sub)) if
                  f.startswith(nm) & f.endswith('.mha')][0]

    res_t1_sa = sitk.GetArrayFromImage(sitk.ReadImage(path_res))
    ssim=np.round(float(str.rsplit(str.rsplit(path_res,'_')[-1],'.mha')[0]),2)
    return res_t1_sa,ssim


if __name__=='__main__':

    inx=45
    T1_RA_SA='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_val_att2/residual_attention2_fold_1/0_vali_result/'
    T1 ='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/0_vali_result/'
    T1_RA='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/denseunet_hybrid_hr_not1_01/0_vali_result/'
    plusT1='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_01/0_vali_result/'
    sub='pp03/W1_HN1/'
    pet='pet_pp03_ASL_'+str(inx)+'_.mha'

    nm = 'asl_pp03_ASL_' + str(inx)
    path_res = [(os.path.join(T1_RA_SA + sub + f)) for f in os.listdir(os.path.join(T1_RA_SA + sub)) if
                f.startswith(nm) & f.endswith('.mha')][0]

    asl_img = sitk.GetArrayFromImage(sitk.ReadImage(path_res))

    res_t1_RA,ssim_t1RA=read_img(sub=sub,path=T1_RA)
    res_t1_RA_SA,ssim_t1RA_SA=read_img(sub=sub,path=T1_RA_SA)
    res_t1,ssim_t1=read_img(sub=sub,path=T1)
    res_plust1,ssim_plust1=read_img(sub=sub,path=plusT1)
    gt=sitk.GetArrayFromImage(sitk.ReadImage(T1_RA+sub+pet))
    wl=.7

    fig, axes = plt.subplots(nrows=2, ncols=3)

    imgplot=axes.flat[0].imshow(asl_img)
    imgplot.set_clim(0.0, 6)
    axes.flat[0].set_title('Input: ASL')
    axes.flat[0].axis('off')

    imgplot = axes.flat[1].imshow(gt)
    imgplot.set_clim(0.0, wl)
    axes.flat[1].set_title('Target')
    axes.flat[1].axis('off')

    imgplot = axes.flat[2].imshow(res_t1)
    imgplot.set_clim(0.0, wl)
    axes.flat[2].set_title('-T1-RA-SA,SSIM=' + str(ssim_t1))
    axes.flat[2].axis('off')

    imgplot = axes.flat[3].imshow(res_plust1)
    imgplot.set_clim(0.0, wl)
    axes.flat[3].set_title('+T1-RA-SA,SSIM=' + str(ssim_plust1))
    axes.flat[3].axis('off')



    imgplot = axes.flat[4].imshow(res_t1_RA)
    imgplot.set_clim(0.0, wl)
    axes.flat[4].set_title('+T1+RA-SA,SSIM='+str(ssim_t1RA))
    axes.flat[4].axis('off')


    imgplot = axes.flat[5].imshow(res_t1_RA_SA)
    imgplot.set_clim(0.0, wl)
    axes.flat[5].set_title('+T1+RA+SA,SSIM=' + str(ssim_t1RA_SA))
    axes.flat[5].axis('off')

    ax = fig.add_subplot(2, 4, 7)
    plt.axis('off')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(imgplot, cax=cbar_ax)




    plt.show()