import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import research_toolbox.rt_stat.harmonization as har



def dv_extract_data(path, demo_file,atlas='bna'):
    demo=pd.read_excel(demo_file)
    new_demo=pd.DataFrame()
    files=os.listdir(path)
    files=sorted(files)
    den_nets=[]
    count_nets=[]
    len_nets=[]
    for file in files:
        data=loadmat(path+file)
        if atlas=='bna':
            den_net=data['brainnetome246_sift_invnodevol_radius2_count_connectivity']
            count_net=data['brainnetome246_sift_radius2_count_connectivity']
            len_net=data['brainnetome246_radius2_meanlength_connectivity']
        elif atlas=='aal':
            den_net=data['aal116_sift_invnodevol_radius2_count_connectivity'][0:90,0:90]
            count_net=data['aal116_sift_radius2_count_connectivity'][0:90,0:90]
            len_net=data['aal116_radius2_meanlength_connectivity'][0:90,0:90]
        elif atlas=='schaefer':
            den_net=data['schaefer400_sift_invnodevol_radius2_count_connectivity']
            count_net=data['schaefer400_sift_radius2_count_connectivity']
            len_net=data['schaefer400_radius2_meanlength_connectivity']
        den_nets.append(den_net)
        count_nets.append(count_net)
        len_nets.append(len_net)
        file=file[0:-4]
        row=demo[demo['id']==file]
        new_demo=pd.concat([new_demo,row],axis=0)
    return np.array(den_nets), np.array(count_nets), np.array(len_nets), new_demo

def extract_retest_data(path,atlas='bna'):
    #path=r'D:\OneDrive\hwj\project\indi_diff\data\hcp_retest_qsi_net'
    files=os.listdir(path)
    files=[file[0:-5] for file in files]
    files=set(files)

    mats_pair=[]
    mats=[]
    len_mats=[]
    for file in files:
        if os.path.exists(path+'\\'+file+'A.mat') and os.path.exists(path+'\\'+file+'B.mat'):
            if atlas=='bna':
                data=loadmat(path+'\\'+file+'A.mat')
                mat1=data['brainnetome246_sift_radius2_count_connectivity']
                len_mat=data['brainnetome246_radius2_meanlength_connectivity']
                data=loadmat(path+'\\'+file+'B.mat')
                mat2=data['brainnetome246_sift_radius2_count_connectivity']
            elif atlas=='aal':
                data=loadmat(path+'\\'+file+'A.mat')
                mat1=data['aal116_sift_radius2_count_connectivity'][0:90,0:90]
                len_mat=data['aal116_radius2_meanlength_connectivity'][0:90,0:90]
                data=loadmat(path+'\\'+file+'B.mat')
                mat2=data['aal116_sift_radius2_count_connectivity'][0:90,0:90]
            elif atlas=='schaefer':
                data=loadmat(path+'\\'+file+'A.mat')
                mat1=data['schaefer400_sift_radius2_count_connectivity']
                len_mat=data['schaefer400_radius2_meanlength_connectivity']
                data=loadmat(path+'\\'+file+'B.mat')
                mat2=data['schaefer400_sift_radius2_count_connectivity']
            mats_pair.append((mat1,mat2))
            mats.append(mat1)
            len_mats.append(len_mat)
    return mats_pair, len_mats

def check_outlier(nets, thre, demo):
    mean_nets=np.mean(nets,axis=0)
    mean_array=mean_nets[np.triu_indices_from(mean_nets,k=1)]
    rval=[]
    for i in range(nets.shape[0]):
        array=nets[i][np.triu_indices_from(mean_nets,k=1)]
        rval.append(np.corrcoef(mean_array,array)[0,1])
    rval=np.asarray(rval)
    sub_outlier=demo.iloc[rval<thre,:]
    return sub_outlier, rval
    

def combat_for_nets_filter(nets,demo):
    '''
    When sparsity of nets is low, combat will increase the sparsity of nets
    '''
    vecs=[]
    for i in range(nets.shape[0]):
        net=np.copy(nets[i])
        vecs.append(net[np.triu_indices_from(net,k=1)])
    vecs=np.array(vecs)
    covars=pd.DataFrame({'center':demo['center'],'age':demo['age'],'sex':demo['sex']})
    categorical_cols = ['sex']
    mask=np.std(vecs,axis=0)>1e-16
    vecs_combat=har.combat(vecs[:,mask],covars,categorical_cols)
    nets_combat=[]
    raw_nets=[]
    k=0
    for vec in vecs_combat:
        tmp_vec=vecs[k]
        tmp_vec[mask]=vec
        net=np.zeros(nets[0].shape)
        raw_net=np.zeros(nets[0].shape)
        net[np.triu_indices_from(net,k=1)]=tmp_vec
        raw_net[np.triu_indices_from(net,k=1)]=tmp_vec
        net+=net.T
        raw_net+=raw_net.T
        nets_combat.append(net)
        raw_nets.append(raw_net)
        k+=1
    nets_combat=np.array(nets_combat)
    raw_nets=np.array(raw_nets)
    return nets_combat, raw_nets

def combat_for_nets(nets,demo):
    '''
    When sparsity of nets is low, combat will increase the sparsity of nets
    '''
    vecs=[]
    for i in range(nets.shape[0]):
        net=np.copy(nets[i])
        vecs.append(net[np.triu_indices_from(net,k=1)])
    vecs=np.array(vecs)
    covars=pd.DataFrame({'center':demo['center'],'age':demo['age'],'sex':demo['sex']})
    categorical_cols = ['sex']
    vecs_combat=har.combat(vecs,covars,categorical_cols)
    nets_combat=[]
    raw_nets=[]
    for vec in vecs_combat:
        net=np.zeros(nets[0].shape)
        raw_net=np.zeros(nets[0].shape)
        net[np.triu_indices_from(net,k=1)]=vec
        raw_net[np.triu_indices_from(net,k=1)]=vec
        net+=net.T
        raw_net+=raw_net.T
        nets_combat.append(net)
        raw_nets.append(raw_net)
    nets_combat=np.array(nets_combat)
    raw_nets=np.array(raw_nets)
    return nets_combat, raw_nets

def demo_stat(demo):
    demo[['age','exe','attention']]=demo[['age','exe','attention']].astype(float)
    print(demo.describe(include='all'))
    print(demo.groupby(['center','sex']).count())
    print(demo.groupby('center').count())
    print(demo[['exe','center']].groupby('center').describe())
    print(demo[['attention','center']].groupby('center').describe())
    print(demo[['age','center']].groupby('center').describe())