import matplotlib.pyplot as plt
import SimpleITK as sitk
import datetime
import os
import shutil
import numpy as np
def read_img(sub,path,ext='.mha'):
    nm = 'res_petpp03_ASL_' + str(inx)
    path_res = [(os.path.join(path + sub + f)) for f in os.listdir(os.path.join(path + sub)) if
                  f.startswith(nm) & f.endswith(ext)][0]

    res_t1_sa = sitk.GetArrayFromImage(sitk.ReadImage(path_res))
    ssim=np.round(float(str.rsplit(str.rsplit(path_res,'_')[-1],ext)[0]),2)
    return res_t1_sa,ssim


if __name__=='__main__':
    # plt.style.use('classic')
    # plt.style.use('white background')
    #

    # ax = plt.axes(axisbg='#E6E6E6')
    # ax = plt.axes(facecolor='#E6E6E6')

    # ax.set_axisbelow(True)

    inx=45#'pp03/W1_HN1/'

    inx=41#'pp03/W6_HY/'
    T1_RA_SA='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_val_att2/residual_attention2_fold_1/0_vali_result/'
    T1 ='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/residual_attention2_UnetSkipAttention1/0_vali_result/'
    T1_RA='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/denseunet_hybrid_hr_not1_01/0_vali_result/'
    plusT1='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/denseunet_hybrid_01/0_vali_result/'
    sub='pp03/W1_HN1/'
    sub='pp03/W6_HY/'
    pet='pet_pp03_ASL_'+str(inx)+'_.nii'
    wl = .65
    ll=0.01

    nm = 'asl_pp03_ASL_' + str(inx)
    path_res = [(os.path.join(T1_RA_SA + sub + f)) for f in os.listdir(os.path.join(T1_RA_SA + sub)) if
                f.startswith(nm) & f.endswith('.mha')][0]

    asl_img = sitk.GetArrayFromImage(sitk.ReadImage(path_res))

    res_t1_RA,ssim_t1RA=read_img(sub=sub,path=T1_RA)
    res_t1_RA_SA,ssim_t1RA_SA=read_img(sub=sub,path=T1_RA_SA)
    res_t1,ssim_t1=read_img(sub=sub,path=T1)
    res_plust1,ssim_plust1=read_img(sub=sub,path=plusT1,ext='.nii')
    gt=sitk.GetArrayFromImage(sitk.ReadImage(T1_RA+sub+pet))

    # plt.figure(facecolor="white")
    fig, axes = plt.subplots(nrows=2, ncols=6,gridspec_kw={'wspace':0.05, 'hspace':.01},facecolor='white')

    imgplot=axes.flat[0].imshow(asl_img)
    imgplot.set_clim(ll, 6)
    axes.flat[0].set_title('Input: ASL')
    axes.flat[0].axis('off')
    axes.flat[0].set_aspect('equal')




    imgplot = axes.flat[1].imshow(gt)
    imgplot.set_clim(ll, wl)
    axes.flat[1].set_title('Target: PET')
    axes.flat[1].axis('off')
    axes.flat[1].set_aspect('equal')


    imgplot = axes.flat[2].imshow(res_t1)
    imgplot.set_clim(ll, wl)
    axes.flat[2].set_title('-T1-RA-SA' )
    axes.flat[2].axis('off')
    axes.flat[2].text(155, 205,str('{:1.2f}'.format(ssim_t1)), fontsize = 12, color = 'red')
    axes.flat[2].set_aspect('equal')



    imgplot = axes.flat[3].imshow(res_t1_RA )
    imgplot.set_clim(ll, wl)
    axes.flat[3].set_title('+T1-RA-SA' )
    axes.flat[3].axis('off')
    axes.flat[3].text(155, 205, str( '{:1.2f}'.format(ssim_t1RA)), fontsize=12, color='red')
    axes.flat[3].set_aspect('equal')




    imgplot = axes.flat[4].imshow(res_plust1)
    imgplot.set_clim(ll, wl)
    axes.flat[4].set_title('+T1+RA-SA')
    axes.flat[4].axis('off')
    axes.flat[4].text(155, 205, str('{:1.2f}'.format(ssim_plust1) ), fontsize=12, color='red')
    axes.flat[4].set_aspect('equal')

    imgplot = axes.flat[5].imshow(res_t1_RA_SA)
    imgplot.set_clim(ll, wl)
    axes.flat[5].set_title('+T1+RA+SA')
    axes.flat[5].axis('off')
    axes.flat[5].text(155, 205, str('{:1.2f}'.format(ssim_t1RA_SA)), fontsize=12, color='red')
    axes.flat[5].set_aspect('equal')
    fig.colorbar(imgplot, ax=axes[0,:], shrink=0.5 ,location='right',cmap='RdBu')

    # ax = fig.add_subplot(2, 4, 7)
    # plt.axis('off')
    # cbar_ax = fig.add_axes([0.15, 0.29, 0.7, 0.03])
    # fig.colorbar(imgplot, cax=cbar_ax, orientation='horizontal', fraction=.1)

    #-------------------------------

    imgplot = axes.flat[6].imshow(asl_img - asl_img)
    imgplot.set_clim(ll, wl)
    # axes.flat[1].set_title('Target: PET')
    axes.flat[6].axis('off')
    axes.flat[6].set_aspect('equal')



    imgplot = axes.flat[7].imshow(gt - gt)
    imgplot.set_clim(ll, wl)
    # axes.flat[1].set_title('Target: PET')
    axes.flat[7].axis('off')
    axes.flat[7].set_aspect('equal')

    imgplot = axes.flat[8].imshow( res_t1-gt)
    imgplot.set_clim(-.2, .2)
    # axes.flat[8].set_title('Error')
    axes.flat[8].axis('off')
    axes.flat[8].set_aspect('equal')
    # axes.flat[8].text(155, 205, str(ssim_t1), fontsize=12, color='red')

    imgplot = axes.flat[9].imshow( res_t1_RA-gt)
    imgplot.set_clim(-.2, .2)
    # axes.flat[9].set_title('Error')
    axes.flat[9].axis('off')
    axes.flat[9].set_aspect('equal')
    # axes.flat[9].text(155, 205, str(ssim_t1RA), fontsize=12, color='red')
    imgplot = axes.flat[10].imshow(res_plust1-gt)
    imgplot.set_clim(-.2, .2)
    # axes.flat[10].set_title('Error')
    axes.flat[10].axis('off')
    axes.flat[10].set_aspect('equal')
    # axes.flat[10].text(155, 205, str(ssim_plust1), fontsize=12, color='red')

    imgplot = axes.flat[11].imshow( res_t1_RA_SA-gt)
    imgplot.set_clim(-.2, .2)
    # axes.flat[11].set_title('Error')
    axes.flat[11].axis('off')
    axes.flat[11].set_aspect('equal')
    # axes.flat[11].text(155, 205, str(ssim_plust1), fontsize=12, color='red')
    # axes.flat[4].colorbar()
    fig.colorbar(imgplot, ax=axes[1, :], shrink=0.5, location='right',cmap='RdBu')


    plt.show()