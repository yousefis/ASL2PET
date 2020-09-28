from openpyxl import load_workbook
import numpy as np
import pandas as pd
def _write_frame_to_new_sheet(path_to_file=None, sheet_name='sheet', data_frame=None):
    book = None
    try:
        book = load_workbook(path_to_file)
    except Exception:
        print('Creating new workbook at %s', path_to_file)
    with pd.ExcelWriter(path_to_file, engine='openpyxl') as writer:
        if book is not None:
            writer.book = book
        data_frame.to_excel(writer, sheet_name, index=False)

def read_print_mean_std_tags(xl_file,tag):
    nc_noskip = xl_file.parse("Sheet1")[tag+"_HC"].to_list()
    hc_nskip = xl_file.parse("Sheet2")[tag+"_NC"].to_list()
    both=nc_noskip+hc_nskip


    return nc_noskip,hc_nskip,both
if __name__=="__main__":
    path = "/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/"

    vali_test=1
    xls_path="/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/stat_res.xlsx"
    NC_ssim=[]
    HC_ssim=[]
    ssim=[]
    NC_mse=[]
    HC_mse=[]
    mse=[]
    NC_psnr=[]
    HC_psnr=[]
    psnr=[]
    for fold in [1,3,4,5,6,7,8]:
        # path_res = path + "cross_val_att2/residual_attention2_fold_" + str(fold) + "/"
        # path_res=path+"cross_validation_unetskipatt/residual_attention2_UnetSkipAttention"+str(fold)+"/"
        path_res=path+"01_cross_validation/plust1_noskippatt_noresidualatt/residual_attention2_plust1_no_resid_noskip_fold_"+str(fold)+"/"
        path_res=path+"01_cross_validation/multitask_not1/denseunet_hybrid_not1_0"+str(fold)+"/"
        path_res=path+"01_cross_validation/single_plust1/single_plust1_fold_"+str(fold)+"/"
        path_res=path+"01_cross_validation/single_plust1_skipatt/single_plust1_skipatt_fold_"+str(fold)+"/"

        if vali_test==1:
            path_res=path_res+"0_vali_result/all_ssim.xlsx"
        else:
            path_res = path_res + "0_test_result/all_ssim.xlsx"

        xl_file = pd.ExcelFile(path_res)
        nc_ssim, hc_ssim, both_ssim = read_print_mean_std_tags(xl_file,tag="SSIM")
        nc_mse, hc_mse, both_mse = read_print_mean_std_tags(xl_file,tag="MSE")
        nc_psnr, hc_psnr, both_psnr = read_print_mean_std_tags(xl_file,tag="PSNR")

        NC_ssim=NC_ssim+nc_ssim
        HC_ssim=HC_ssim+hc_ssim
        ssim=ssim+both_ssim

        NC_mse = NC_mse + nc_mse
        HC_mse = HC_mse + hc_mse
        mse = mse + both_mse

        NC_psnr = NC_psnr + nc_psnr
        HC_psnr = HC_psnr + hc_psnr
        psnr = psnr + both_psnr

    print(np.average(NC_ssim).round(2),
          np.std(NC_ssim).round(2))
    print(np.average(NC_mse).round(2),
          np.std(NC_mse).round(2))
    print(np.average(NC_psnr).round(1),
          np.std(NC_psnr).round(1))


    print(np.average(HC_ssim).round(2),
          np.std(HC_ssim).round(2))
    print(np.average(HC_mse).round(2),
          np.std(HC_mse).round(2))
    print(np.average(HC_psnr).round(1),
          np.std(HC_psnr).round(1))

    print(np.average(ssim).round(2),
          np.std(ssim).round(2))
    print(np.average(mse).round(2),
          np.std(mse).round(2))
    print(np.average(psnr).round(1),
          np.std(psnr).round(1))
















