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
    dfs_noskip = xl_file.parse("Sheet1")[tag+"_HC"].to_list() + xl_file.parse("Sheet2")[tag+"_NC"].to_list()

    print(np.average(xl_file.parse("Sheet1")[tag+"_HC"].to_list()).round(2),
          np.std(xl_file.parse("Sheet1")[tag+"_HC"].to_list()).round(2))
    print(np.average(xl_file.parse("Sheet2")[tag+"_NC"].to_list()).round(2),
          np.std(xl_file.parse("Sheet2")[tag+"_NC"].to_list()).round(2))
    print(np.average(dfs_noskip).round(2), np.std(dfs_noskip).round(2))
    return dfs_noskip
if __name__=="__main__":
    path = "/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/"

    vali_test=1
    xls_path="/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/stat_res.xlsx"
    for fold in [8]:#range(3,7)]:
        skip=path+"cross_validation_unetskipatt/residual_attention2_UnetSkipAttention"+str(fold)+"/"
        noskip=path+"cross_val_att2/residual_attention2_fold_"+str(fold)+"/"
        if vali_test==1:
            skip=skip+"0_vali_result/all_ssim.xlsx"
            noskip=noskip+"0_vali_result/all_ssim.xlsx"
        else:
            skip = skip + "0_test_result/all_ssim.xlsx"
            noskip = noskip + "0_test_result/all_ssim.xlsx"

        xl_file = pd.ExcelFile(noskip)
        dfs_noskip = read_print_mean_std_tags(xl_file,tag="SSIM")
        _ = read_print_mean_std_tags(xl_file,tag="MSE")
        _ = read_print_mean_std_tags(xl_file,tag="PSNR")


        xl_file = pd.ExcelFile(skip)
        dfs_skip = read_print_mean_std_tags(xl_file, tag="SSIM")
        _ = read_print_mean_std_tags(xl_file, tag="MSE")
        _ = read_print_mean_std_tags(xl_file, tag="PSNR")



        df = pd.DataFrame.from_dict({'with_skip': dfs_skip,
                                     'without_skip': dfs_noskip,
                                     })

        # _write_frame_to_new_sheet(path_to_file=xls_path, sheet_name='fold'+str(fold), data_frame=df)


        print(1)






