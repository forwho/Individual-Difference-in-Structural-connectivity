import os
import util_data_preprocess as udp
import util_data_stat as uds
import numpy as np
from numpy.matlib import repmat
import pandas as pd
import pickle
import dill
import research_toolbox.individual_difference.dv as ridv
import research_toolbox.individual_difference.dv_stat as ridvs
from research_toolbox import rt_stat
from scipy import stats
from research_toolbox.visulization.brain_surface import vis_wb
from research_toolbox.visulization.stat import stat as rtv_stat
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from research_toolbox.gene_set import gene_preprocess as gprep
from research_toolbox.gene_set import gene_stat as gstat

def main_1_extract(atlas='bna'):
    den_nets, count_nets, len_nets, demo=udp.dv_extract_data('data/nets/old/','data/demo/all_data_cog.xlsx',atlas)
    np.savez('data/1_preprocess/nets_demo_%s.npz' % atlas,den_nets=den_nets,count_nets=count_nets,len_nets=len_nets,demo=demo)

def main_1_extract_hcpy(atlas='bna'):
    den_nets, count_nets, len_nets, demo=udp.dv_extract_data('data/nets/hcpy/','data/demo/hcp-y.xlsx',atlas)
    np.savez('data/1_preprocess/hcpy_nets_demo_%s.npz' % atlas,den_nets=den_nets,count_nets=count_nets,len_nets=len_nets,demo=demo)

def main_1_extract_retest(atlas='bna'):
    mats_pair, len_nets=udp.extract_retest_data('data/nets/hcp_retest/',atlas)
    np.savez('data/1_preprocess/retest_nets_demo_%s.npz' % atlas,mats_pair=mats_pair,len_nets=len_nets)

def main_1_preprocess(atlas='bna',edge_thre=-10000, corr_thre=0.85):
    # filter subjects whose age less than 30 or more than 68
    data=np.load('data/1_preprocess/nets_demo_%s.npz' % atlas,allow_pickle=True)
    count_nets=data['count_nets']
    len_nets=data['len_nets']
    demo=pd.DataFrame(data['demo'],columns=['num','id','age','sex','center','exe','attention'])
    count_nets=count_nets[np.bitwise_and(demo['age']>30, demo['age']<=68)]
    len_nets=len_nets[np.bitwise_and(demo['age']>30, demo['age']<=68)]
    demo=demo.loc[np.bitwise_and(demo['age']>30, demo['age']<=68),:]

    # combat count nets across HCA, Cam-CAN and BABRI
    combat_count_nets, raw_count_nets=udp.combat_for_nets(count_nets,demo)
    len_nets[combat_count_nets<edge_thre]=0
    combat_count_nets[combat_count_nets<edge_thre]=0

    # filter subjects whose nets far away from mean nets
    sub_outlier, rval=udp.check_outlier(combat_count_nets, corr_thre, demo)
    combat_count_nets=combat_count_nets[rval>=corr_thre]
    len_nets=len_nets[rval>=corr_thre]
    demo=demo.loc[rval>=corr_thre,:]
    np.savez('data/1_preprocess/nets_demo_%s_%02d_%.2f_preprocess.npz' % (atlas, edge_thre, corr_thre),count_nets=combat_count_nets,len_nets=len_nets,demo=demo)

def main_2_dv(atlas='bna',edge_thre=0, corr_thre=0.85):
    # read data of networks
    aging_data=np.load('data/1_preprocess/nets_demo_%s_%02d_%.2f_preprocess.npz' % (atlas, edge_thre, corr_thre),allow_pickle=True)
    demo=pd.DataFrame(aging_data['demo'],columns=['num','id','age','sex','center','exe','attention'])
    retest_data=np.load('data/1_preprocess/retest_nets_demo_%s.npz' % atlas,allow_pickle=True)
    hcpy_data=np.load('data/1_preprocess/hcpy_nets_demo_%s.npz' % atlas,allow_pickle=True)

    # calculate the dvs
    aging_dv=ridv.batch_dv(aging_data['count_nets'],0)
    hca_dv=ridv.batch_dv(aging_data['count_nets'][demo['center']==1])
    camcan_dv=ridv.batch_dv(aging_data['count_nets'][demo['center']==2])
    babri_dv=ridv.batch_dv(aging_data['count_nets'][demo['center']==3])

    hcpy_dv=ridv.batch_dv(hcpy_data['count_nets'],0)
    intra_dv=ridv.intra_dv_conn(retest_data['mats_pair'])
    inter_dv=ridv.batch_dv(retest_data['mats_pair'][:,0,:,:])

    wind_dv, mean_age=ridv.wind_indv_conn(demo['age'].to_numpy(),aging_data['count_nets'],10,1)

    # regress out intraindividual differences from interindividual differences
    slope, intercept, r_value, p_value, std_err=stats.linregress(inter_dv,intra_dv)
    aging_dv=aging_dv-slope*intra_dv
    hcpy_dv=hcpy_dv-slope*intra_dv
    hca_dv=hca_dv-slope*intra_dv
    camcan_dv=camcan_dv-slope*intra_dv
    babri_dv=babri_dv-slope*intra_dv

    for i in range(wind_dv.shape[0]):
        wind_dv[i]=wind_dv[i]-slope*intra_dv

    # save data of dvs
    np.save('data/2_dv/aging_dv.npy',aging_dv)
    np.save('data/2_dv/hca_dv.npy',hca_dv)
    np.save('data/2_dv/camcan_dv.npy',camcan_dv)
    np.save('data/2_dv/babri_dv.npy',babri_dv)
    np.save('data/2_dv/hcpy_dv.npy',hcpy_dv)
    np.save('data/2_dv/intra_dv.npy',intra_dv)
    np.savez('data/2_dv/wind_dv_age.npz', wind_dv=wind_dv, mean_age=mean_age)

def main_2_plot_1_s1_s2(atlas='bna'):
    # print statistics of demographic information
    data=np.load('data/1_preprocess/nets_demo_%s.npz' % atlas,allow_pickle=True)
    demo=pd.DataFrame(data['demo'],columns=['num','id','age','sex','center','exe','attention'])
    udp.demo_stat(demo)

    # read dv data and save them in dscaler.nii and nii.gz files
    aging_dv=np.load('data/2_dv/aging_dv.npy')
    hca_dv=np.load('data/2_dv/hca_dv.npy')
    camcan_dv=np.load('data/2_dv/camcan_dv.npy')
    babri_dv=np.load('data/2_dv/babri_dv.npy')
    hcpy_dv=np.load('data/2_dv/babri_dv.npy')
    intra_dv=np.load('data/2_dv/intra_dv.npy')
    wind_data=np.load('data/2_dv/wind_dv_age.npz')
    wind_dv=wind_data['wind_dv']
    mean_age=wind_data['mean_age']

    vis_wb.array2dscalar(aging_dv[0:210],'data/figure_data/figure_1/aging_dv.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(aging_dv,'data/figure_data/figure_1/aging_dv.nii.gz')
    vis_wb.array2dscalar(hcpy_dv[0:210],'data/figure_data/figure_1/hcpy_dv.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(hcpy_dv,'data/figure_data/figure_1/hcpy_dv.nii.gz')

    vis_wb.array2dscalar(intra_dv[0:210],'data/figure_data/figure_s1/intra_dv.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(intra_dv,'data/figure_data/figure_s1/intra_dv.nii.gz')

    vis_wb.array2dscalar(hca_dv[0:210],'data/figure_data/figure_s2/hca_dv.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(hca_dv,'data/figure_data/figure_s2/hca_dv.nii.gz')
    vis_wb.array2dscalar(camcan_dv[0:210],'data/figure_data/figure_s2/camcan_dv.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(camcan_dv,'data/figure_data/figure_s2/camcan_dv.nii.gz')
    vis_wb.array2dscalar(babri_dv[0:210],'data/figure_data/figure_s2/babri_dv.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(babri_dv,'data/figure_data/figure_s2/babri_dv.nii.gz')

    # plot figure 1C and print p values
    tmp_dv=(aging_dv-np.mean(aging_dv))/np.std(aging_dv)
    res=stats.kstest(tmp_dv,'norm',N=tmp_dv.shape[0])
    print('Normative test for aging dv: %.6f' % res.pvalue)

    all_dvs=np.asarray([hcpy_dv,aging_dv,hca_dv,camcan_dv,babri_dv])
    rvals=np.corrcoef(all_dvs)
    sns.set_theme(style="white")
    mask = np.triu(np.ones_like(rvals, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap=mpl.colormaps['turbo']
    sns.heatmap(rvals, mask=mask, cmap=cmap, vmax=1, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    f.savefig('data/figure_data/figure_1/dv_site_corr.tiff',dpi=300)

    sitename=['HCPY','Aging','HCA','CAMCAN','BABRI']
    rval, pval, xrand=rt_stat.space_correction.spatial_autocorrelation_correction(hcpy_dv,aging_dv,xrand=None,repeatn=1000,method='pearson',atlas='bna')
    for i in range(all_dvs.shape[0]):
        for j in range(i+1,all_dvs.shape[0]):
            rval, pval, xrand=rt_stat.space_correction.spatial_autocorrelation_correction(all_dvs[i],all_dvs[j],xrand,repeatn=1000,method='pearson',atlas='bna')
            print("%s-%s: rval is %.3f and pval is %.6f" % (sitename[i],sitename[j],rval,pval))

    # plot figure 1D-F and print statisticc data
    rvals, pvals, thre=ridvs.dv_age_wind(wind_dv,mean_age,'spearman')
    tmp_age=(mean_age-np.mean(mean_age))/np.std(mean_age)
    res=stats.kstest(tmp_age,'norm',N=tmp_age.shape[0])
    print('Normative test for mean age: %.6f' % res.pvalue)

    rvals[pvals>thre]=np.nan
    vis_wb.array2dscalar(rvals[0:210],'data/figure_data/figure_1/corr_dv_age.dscalar.nii',mode=atlas)
    vis_wb.save_subcortical(rvals,'data/figure_data/figure_1/corr_dv_age.nii.gz')

    mean_dvs=np.mean(wind_dv,axis=1)
    std_dvs=np.std(wind_dv,axis=1)
    print('The correlation between mean dvs and age:',stats.spearmanr(mean_dvs,mean_age))
    print('The correlation between std dvs and age:',stats.spearmanr(std_dvs,mean_age))

    scatter_kws={'alpha':1,'s':70,'color':'#E6550D','linewidths':1,'edgecolors':'000000','alpha':0.5}
    line_kws={'linewidth':5,'color':'#E6550D'}
    rtv_stat.corr_plot(mean_age,mean_dvs,'Mean age', 'Mean variation',1, scatter_kws, line_kws, 'data/figure_data/figure_1/corr_meanvar_age.tiff')
    rtv_stat.corr_plot(mean_age,std_dvs,'Mean age', 'Standard deviation of variation',1, scatter_kws, line_kws, 'data/figure_data/figure_1/corr_stdvar_age.tiff')

def main_2_plot_2():

    # plot figure 2B
    aging_dv=np.load('data/2_dv/aging_dv.npy')
    dv_yeo=ridvs.yeo_stat(aging_dv,is_print=True)
    yeo_name=np.asarray(['VN', 'SM', 'DAN', 'VAN', 'Lim', 'FPN', 'DMN','Sub'])
    index=np.argsort(dv_yeo)
    dv_yeo=dv_yeo[index]
    yeo_name=yeo_name[index]
    color=np.asarray(['#822956', '#654765', '#70709D', '#29FF29', '#C3F22D', '#FFA129', '#FF2929', '#D684BD'])
    color=color[index]
    rtv_stat.bar_plot(yeo_name, dv_yeo, color, False, 'data/figure_data/figure_2/dv_yeo_aging.tiff')

    # plot figure 2C and print statistic data
    wind_data=np.load('data/2_dv/wind_dv_age.npz')
    wind_dv_yeos=ridvs.yeo_stat_wind(wind_data['wind_dv'])
    print(yeo_name)
    print(ridvs.dv_age_wind(wind_dv_yeos,wind_data['mean_age'],'spearman'))

    xval=wind_data['mean_age']
    yval=wind_dv_yeos
    nanindex=np.isnan(xval)
    xval=xval[np.logical_not(nanindex)]
    yval=yval[np.logical_not(nanindex)]
    for i in range(8):
        scatter_kws={'alpha':1,'s':10,'edgecolors':'none','color':color[i]}
        line_kws={'linewidth':3,'color':color[i]}
        scatter_kws={'alpha':1,'s':70,'color':color[i],'linewidths':1,'edgecolors':'000000','alpha':0.5}
        line_kws={'linewidth':5,'color':color[i]}
        rtv_stat.corr_plot(xval,yval[:,i],'Mean age', 'Mean variation', 1,scatter_kws, line_kws, 'data/figure_data/figure_2/yeo_wind_%s_filter.tiff' % yeo_name[i])

def main_3_plot_3(atlas='bna',edge_thre=0,corr_thre=0.85):

    # plot 3 A,B,D,E
    aging_dv=np.load('data/2_dv/aging_dv.npy')
    results, revo_data, val_datas=ridvs.corr_other_maps(aging_dv)

    map_names=['CBF', 'Myelin', 'Gradients', 'HistGradients_G1','HistGradients_G2','L1Thickness','L2Thickness','L3Thickness','L4Thickness','L5Thickness','L6Thickness']
    color=sns.color_palette()
    indexes=[1,4]
    for index in indexes:
        vis_wb.array2dscalar(val_datas[index][0:210],'data/figure_data/figure_3/%s_map.dscalar.nii' % (map_names[index]),mode='bna')
        scatter_kws={'alpha':1,'s':70,'edgecolors':'none','color':color[index],'linewidths':1,'edgecolors':'000000'}
        line_kws={'linewidth':5,'color':color[index]}
        rtv_stat.corr_plot(aging_dv[0:210],val_datas[index],'Individual variability', map_names[index],1,scatter_kws, line_kws, 'data/figure_data/figure_3/corr_dv_%s.tiff' % map_names[index])

    # plot 3 C,F
    aging_data=np.load('data/1_preprocess/nets_demo_%s_%02d_%.2f_preprocess.npz' % (atlas, edge_thre, corr_thre),allow_pickle=True)
    results, mean_strength, short_percent=ridvs.len_nets_stat(aging_dv,aging_data['count_nets'],aging_data['len_nets'],2,method='number',mode='network',is_print=True)
    vis_wb.array2dscalar(mean_strength[0,0:210],'data/figure_data/figure_3/short_strength_map.dscalar.nii',mode='bna')
    scatter_kws={'alpha':1,'s':70,'edgecolors':'none','color':color[0],'linewidths':1,'edgecolors':'000000'}
    line_kws={'linewidth':5,'color':color[0]}
    rtv_stat.corr_plot(aging_dv,mean_strength[0],'Individual variability', 'Mean strength of short edges', 1,scatter_kws, line_kws, 'data/figure_data/figure_3/corr_dv_len.tiff')

def main_3_len_replication(atlas='bna',edge_thre=0,corr_thre=0.85):
    # plot 3 C,F
    aging_data=np.load('data/1_preprocess/nets_demo_%s_%02d_%.2f_preprocess.npz' % (atlas, edge_thre, corr_thre),allow_pickle=True)
    dvs=[]
    edge_thres=[10,20,30,40,50,60,70,80,100]
    for i in range(len(edge_thres)):
        dvs.append(ridv.batch_dv(aging_data['count_nets'],edge_thres[i]))
    color=sns.color_palette()
    for i in range(len(edge_thres)):
        count_nets=np.copy(aging_data['count_nets'])
        len_nets=np.copy(aging_data['len_nets'])
        len_nets[count_nets<edge_thres[i]]=0
        count_nets[count_nets<edge_thres[i]]=0

        results, revo_data, val_datas=ridvs.corr_other_maps(dvs[i])

        results, mean_strength, short_percent=ridvs.len_nets_stat(dvs[i],count_nets,len_nets,2,method='number',mode='network',is_print=True)
        vis_wb.array2dscalar(mean_strength[0,0:210],'data/figure_data/figure_3/short_strength_map_%02d.dscalar.nii' % edge_thres[i],mode='bna')
        scatter_kws={'alpha':1,'s':70,'edgecolors':'none','color':color[0],'linewidths':1,'edgecolors':'000000'}
        line_kws={'linewidth':5,'color':color[0]}
        rtv_stat.corr_plot(dvs[i],mean_strength[0],'Individual variability', 'Mean strength of short edges', 1,scatter_kws, line_kws, 'data/figure_data/figure_3/corr_dv_len_%02d.tiff' % edge_thres[i])
        np.save('data/2_dv/aging_dv_%02d.npy' % edge_thres[i],dvs[i])

def main_4_gene_pls():
    gene_data=pd.read_csv('data/4_gene_expression/brain_genes_exp.csv')
    dv=np.load('data/2_dv/aging_dv.npy')
    samples=np.load('data/4_gene_expression/samples.npy')
    gene_weights, xscores, r2, xrot, yloadings=gstat.gene_stat_run(gene_data,dv,samples)
    print('r2 is %.3f' % r2)
    xscores=np.squeeze(xscores)
    color_map=sns.color_palette("Set2")
    scatter_kws={'alpha':1,'s':70,'edgecolors':'none','color':color_map[2],'linewidths':1,'edgecolors':'000000'}
    line_kws={'linewidth':5,'color':color_map[2]}
    dv=dv[0:246:2]
    rtv_stat.corr_plot(dv,xscores,'Individual variability', 'Gene scores', 1,scatter_kws, line_kws, 'data/figure_data/figure_4/corr_gene_dv.tiff')

    plot_data=np.zeros(246)
    plot_data[0:246:2]=xscores
    vis_wb.array2dscalar(plot_data[0:210],'data/figure_data/figure_4/gene_score.dscalar.nii',mode='bna')

    gene_weights.to_csv('data/4_gene_expression/true_gene_weights.csv', index=False)
    np.savez('data/4_gene_expression/psl_results.npz', xscores=xscores, r2=r2, xrot=xrot, yloadings=yloadings)

def main_4_gene_go_enrich():
    gene_weights=pd.read_csv('data/4_gene_expression/true_gene_weights.csv')
    go_weights=gstat.go_cate_weights(gene_weights,0,2000,0.5)

    files=os.listdir('data/4_gene_expression/gene_set_random_weight_False_brain_pls_0.50')
    files.sort()
    perm_pos_weights=[]
    perm_neg_weights=[]
    for file in files:
        data=pd.read_csv('data/4_gene_expression/gene_set_random_weight_False_brain_pls_0.50/'+file)
        pos_weight=data['pos_weight']
        neg_weight=data['neg_weight']
        perm_pos_weights.append(pos_weight.to_numpy())
        perm_neg_weights.append(neg_weight.to_numpy())
    perm_pos_weights=np.asarray(perm_pos_weights)
    perm_neg_weights=np.asarray(perm_neg_weights)

    go_weights, pos_thre, neg_thre=gstat.cate_pvalues(go_weights, perm_pos_weights, perm_neg_weights)
    go_weights.to_csv('data/4_gene_expression/go_weights.csv', index=False)

def main_4_cell_enrich():
    gene_weights=pd.read_csv('data/4_gene_expression/true_gene_weights.csv')
    cell_weights=gstat.gene_cell(gene_weights,0.5)
    cell_weights.to_csv('data/4_gene_expression/cell_weights.csv', index=False)

    positive_weight= cell_weights['pos_ratio']
    negative_weight= cell_weights['neg_ratio']
    color_map=sns.color_palette('pastel')
    name=np.asarray(['Astro', 'Endo', 'Micro', 'Neuro-Ex', 'Neuro-In', 'OPC', 'Oligo'])
    rtv_stat.bar_plot(name, positive_weight, color_map[0:5], False, 'data/figure_data/figure_4/pos_cell_weights.tiff')
    rtv_stat.bar_plot(name, negative_weight, color_map[0:5], False, 'data/figure_data/figure_4/neg_cell_weights.tiff')

def main_4_dev_analysis():
    pls_result=np.load('data/4_gene_expression/psl_results.npz')
    gene_weights=pd.read_csv('data/4_gene_expression/true_gene_weights.csv')
    xrot_pd=pd.DataFrame({'gene_symbol':gene_weights.loc[:,'gene_symbol'].to_numpy(),'xrot':pls_result['xrot']})
    deve_exp=gprep.develop_gene_score(xrot_pd)
    dv=np.load('data/2_dv/aging_dv.npy')
    deve_dv=uds.map_dv2brod(dv)
    deve_exp=deve_exp[-1::-1]
    plot_data=pd.DataFrame({'x':['Adult','Adolescent','Child','Infant','Foetus']*16,'y':np.reshape(deve_exp,80,order='F'),'absy':np.abs(np.reshape(deve_exp,80,order='F')),'Region':np.reshape(repmat(np.asarray(['A1C','AMY','CBC','DFC','HIP','IPC','ITC','M1C','MD','MFC','OFC','S1C','STC','STR','V1C','VFC']),5,1),80,order='F')})
    sns.set_palette('tab20')
    sns.set_theme(style="whitegrid")
    g=sns.relplot(x='Region',y='x',hue='y',data=plot_data,size='absy',palette='vlag',edgecolor='.7',sizes=(100,600),height=5,aspect=3)
    g.savefig('data/figure_data/figure_4/deve_exp.tiff',dpi=300)

    deve_exp=deve_exp[-1::-1]
    deve_exp=deve_exp[:,deve_dv>0]
    deve_dv=deve_dv[deve_dv>0]
    color_map=sns.color_palette('Set2')
    for i in range(5):
        scatter_kws={'alpha':1,'s':70,'edgecolors':'none','color':color_map[i],'linewidths':1,'edgecolors':'000000'}
        line_kws={'linewidth':5,'color':color_map[i]}
        rtv_stat.corr_plot(deve_dv,np.squeeze(deve_exp[i,:]),'Individual variability', 'Gene scores', 1,scatter_kws, line_kws, 'data/figure_data/figure_4/deve_exp_dv_corr_%02d' % i)

def main_5_ordered_2_fold_cv():
    data=np.load('data/1_preprocess/nets_demo_bna.npz',allow_pickle=True)
    demo=pd.DataFrame(data['demo'],columns=['num','id','age','sex','center','exe','attention'])
    count_nets=data['count_nets'][np.bitwise_and(demo['age']>30, demo['age']<=68)]
    demo=demo.loc[np.bitwise_and(demo['age']>30, demo['age']<=68),:]

    combat_count_nets, raw_count_nets=udp.combat_for_nets(count_nets,demo)
    combat_count_nets[combat_count_nets<0]=0
    sub_outlier, rval=udp.check_outlier(combat_count_nets, 0.85, demo)
    count_nets=count_nets[rval>=0.85]
    demo=demo.loc[rval>=0.85,:]

    features=[]
    for net in count_nets:
        features.append(net[np.triu_indices_from(net,k=1)])
    features=np.asarray(features)

    best_l1_lambdas, best_rvals, best_maes, best_models, best_trainers, best_trans, test_hats, test_labels=uds.nested_2fold_cv_train(demo,features,'exe',fold_n=2,is_act=True,norm_method='norm',repeat_num=10,test_flag=False,is_shuffle=False,is_rand=False)

    with open('data/5_cognitive_prediction/exe_pred_data.txt', 'wb') as f:
        pickle.dump((best_l1_lambdas,best_rvals,best_maes,test_hats,test_labels),f)
    with open('data/5_cognitive_prediction/exe_best_models.txt', 'wb') as f:
        dill.dump(best_models,f)
    with open('data/5_cognitive_prediction/exe_best_tainers.txt', 'wb') as f:
        dill.dump(best_trainers,f)
    with open('data/5_cognitive_prediction/exe_best_trans.txt', 'wb') as f:
        dill.dump(best_trans,f)



if __name__=='__main__':

    main_5_ordered_2_fold_cv()




    
    



    



    
    




