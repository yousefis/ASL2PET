import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def hgrid(axes,top,bottom,major_gap,minor_gap,axis):
    axes.set_ylim(bottom, top)
    major_ticks = np.arange(bottom, top, major_gap)
    minor_ticks = np.arange(bottom, top, minor_gap)
    axes.set_yticks(major_ticks)
    axes.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    # axes[0].grid(which='both')
    # Or if you want different settings for the grids:
    axes.grid(which='minor', alpha=0.05,axis=axis)
    axes.grid(which='major', alpha=0.5,axis=axis)
    plt.setp(axes.get_xticklabels(), visible=False)




def fill_pb_color(bplots,colors,axes):
    for bplot in bplots:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

def plot_bp(axes,title,data):
    bplot = axes.boxplot(data,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             # labels=cnn_tags,
                             showmeans=True,
                             )  # will be used to label x-ticks
    axes.set_title(title)
    return bplot


def cumulative_dice(axes,cnn_tags,data,title):
    vec = np.zeros([len(data),11])
    # j=0
    for i in reversed(range(0, 11, 1)):
        for dc in range(len(data)):
            vec[ dc,i] = sum(data[dc] < (i / 10))
        # j=j+1
    y_labels = [ '0.0$\leq$', '0.1$\leq$','0.2$\leq$','0.3$\leq$','0.4$\leq$', '0.5$\leq$',  '0.6$\leq$',
                 '0.7$\leq$','0.8$\leq$','0.9$\leq$', '1.0$\leq$'
                ]
    df = pd.DataFrame({
                       # cnn_tags[0]: vec[0],
                       # cnn_tags[1]: vec[1],
                       cnn_tags[2]: vec[2],
                       cnn_tags[3]: vec[3],
                       cnn_tags[4]: vec[4],
                       cnn_tags[5]: vec[5],
                       # cnn_tags[6]: vec[6],
                       }, index=y_labels)
    # df.plot(ax=axes)
    # axes = df.plot.barh(color=colors)
    bplot=df.plot(kind='bar', legend=False, ax=axes,color=colors,title=title,rot=-300)
    return bplot

def read_xls(xls_path,results,parent_pth,fields,sheet_name):
    ssim=[]
    mse=[]
    psnr=[]
    for xp in xls_path:
        p=parent_pth+xp+results+'all_ssim.xlsx'
        print(parent_pth+xp+results)
        # sheet_name = 'Sheet1'
        xl_file = pd.ExcelFile(p)
        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}
        ssim.append(dfs[sheet_name][fields[0]])
        mse.append(dfs[sheet_name][fields[1]])
        psnr.append(dfs[sheet_name][fields[2]])
    return ssim,mse,psnr
#this file shows the boxplots for the journal paper
def print_ave_std(i,ssim_hc,ssim_nc,psnr_hc,psnr_nc,mse_hc,mse_nc):
    print(np.average(np.array(ssim_hc[i].append(ssim_nc[i]))).round(2), np.std(np.array(ssim_hc[i].append(ssim_nc[i]))).round(2))
    print(np.average(np.array(mse_hc[i].append(mse_nc[i]))).round(3), np.std(np.array(mse_hc[i].append(mse_nc[i]))).round(3))
    print(np.average(np.array(psnr_hc[i].append(psnr_nc[i]))).round(2), np.std(np.array(psnr_hc[i].append(psnr_nc[i]))).round(2))

if __name__=='__main__':
    parent_pth='/exports/lkeb-hpc/syousefi/Code/Log_asl_pet/rest/cross_validation_unetskipatt/'
    xls_path=[
              # 'residual_attention2_fold_1/',
              # 'residual_attention2_fold_2/',
              # 'residual_attention2_UnetSkipAttention1/',
              # 'residual_attention2_UnetSkipAttention2/',
              'residual_attention2_UnetSkipAttention3/',
              'residual_attention2_UnetSkipAttention4/',
              'residual_attention2_UnetSkipAttention5/',
              'residual_attention2_UnetSkipAttention6/',
              ]
    cnn_tags=['F1',
              'F2',
              'F3',
              'F4',
              'F5',
              'F6',
              ]
    test_vali=1
    if test_vali==0:
        results='result/'
    else:
        results='0_vali_result/'

    #=============================================

    #read xls files and fill vectors
    ssim_hc,mse_hc,psnr_hc= read_xls(xls_path, results, parent_pth, fields= ['SSIM_HC','MSE_HC','PSNR_HC'],sheet_name='Sheet1')
    ssim_nc, mse_nc, psnr_nc= read_xls(xls_path, results, parent_pth, fields= ['SSIM_NC','MSE_NC','PSNR_NC'],sheet_name='Sheet2')
    #plot boxplots
    col=3
    fig, axes = plt.subplots(nrows=1, ncols=col, figsize=(9, 4))
    bplot1= plot_bp(axes[0], 'SSIM_HC',ssim_hc)
    bplot2= plot_bp(axes[1], 'SSIM_NC',ssim_nc)
    bplot3= plot_bp(axes[2], 'SSIM',[
                                        ssim_hc[0].append(ssim_nc[0]),
                                          ssim_hc[1].append(ssim_nc[1]),
                                          ssim_hc[2].append(ssim_nc[2]),
                                          ssim_hc[3].append(ssim_nc[3]),
                                          # ssim_hc[4].append(ssim_hc[4]),
                                     # ssim_hc[5].append(ssim_hc[5])
                    ])
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen','orchid','tomato','hotpink','cyan']
    fill_pb_color((bplot1,bplot2,bplot3), colors, axes[1])
    hgrid(axes[0],top=1,bottom=0,major_gap=.1,minor_gap=0.05,axis='y')
    hgrid(axes[1],top=1,bottom=0,major_gap=.1,minor_gap=0.05,axis='y')
    hgrid(axes[2],top=1,bottom=0,major_gap=.1,minor_gap=0.05,axis='y')


    # =============================================
    print_ave_std(0, ssim_hc, ssim_nc, psnr_hc, psnr_nc, mse_hc, mse_nc)
    print_ave_std(1, ssim_hc, ssim_nc, psnr_hc, psnr_nc, mse_hc, mse_nc)
    print_ave_std(2, ssim_hc, ssim_nc, psnr_hc, psnr_nc, mse_hc, mse_nc)
    print_ave_std(3, ssim_hc, ssim_nc, psnr_hc, psnr_nc, mse_hc, mse_nc)

    # =============================================
    # plt.figure()
    col = 3
    fig, axes = plt.subplots(nrows=1, ncols=col, figsize=(9, 4))
    bplot1 = plot_bp(axes[0], 'MSE_HC', mse_hc)
    bplot2 = plot_bp(axes[1], 'MSE_NC', mse_nc)
    bplot3 = plot_bp(axes[2], 'MSE', [mse_hc[0].append(mse_nc[0]),
                                       mse_hc[1].append(mse_nc[1]),
                                       mse_hc[2].append(mse_nc[2]),
                                       mse_hc[3].append(mse_nc[3]),
                                       # mse_hc[4].append(mse_hc[4])
                                      ])
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'orchid', 'tomato', 'hotpink', 'cyan']
    fill_pb_color((bplot1, bplot2, bplot3), colors, axes[1])
    hgrid(axes[0], top=.2, bottom=0, major_gap=0.05, minor_gap=0.05, axis='y')
    hgrid(axes[1], top=.2, bottom=0, major_gap=0.05, minor_gap=0.05, axis='y')
    hgrid(axes[2], top=.2, bottom=0, major_gap=0.05, minor_gap=0.05, axis='y')

    # =============================================
    # plt.figure()
    col = 3
    fig, axes = plt.subplots(nrows=1, ncols=col, figsize=(9, 4))
    bplot1 = plot_bp(axes[0], 'PSNR_HC', psnr_hc)
    bplot2 = plot_bp(axes[1], 'PSNR_NC', psnr_nc)
    bplot3 = plot_bp(axes[2], 'PSNR', [psnr_hc[0].append(psnr_nc[0]),
                                      psnr_hc[1].append(psnr_nc[1]),
                                      psnr_hc[2].append(psnr_nc[2]),
                                      psnr_hc[3].append(psnr_nc[3]),
                                      # psnr_hc[4].append(psnr_hc[4])
                                       ])
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'orchid', 'tomato', 'hotpink', 'cyan']
    fill_pb_color((bplot1, bplot2, bplot3), colors, axes[1])
    hgrid(axes[0], top=40, bottom=0, major_gap=10, minor_gap=5, axis='y')
    hgrid(axes[1], top=40, bottom=0, major_gap=10, minor_gap=5, axis='y')
    hgrid(axes[2], top=40, bottom=0, major_gap=10, minor_gap=5, axis='y')



    # =============================================

    # axes[ 2].legend(
    #     [bplot2["boxes"][0], bplot2["boxes"][1], bplot2["boxes"][2], bplot2["boxes"][3], bplot2["boxes"][4],
    #      bplot2["boxes"][5], bplot2["boxes"][6]],
    #     cnn_tags, loc='right', bbox_to_anchor=(1.1, -0.3),
    #     fancybox=True, shadow=True, ncol=5)

    # fig.subplots_adjust(bottom=0.3)  # or whatever

    plt.show()
    print(2)
    # adding horizontal grid lines

