import numpy as np
import nibabel as nib
from pygam import LinearGAM, s
from scipy import stats
from research_toolbox import rt_stat
import pandas as pd
import research_toolbox
from research_toolbox.gene_set import gene_stat as gstat
from research_toolbox.machine_learning.Datasets import MyDataSets
from research_toolbox.machine_learning.ElasticNetwork import train
import pickle
import dill

def map_dv2brod(dv):
    dv_bna_index=[[71],[211,213],[],[5, 23, 15], [215,217],[129, 139, 141], [83, 89, 93, 95, 99, 101, 103], [67, 161], [231,233,235,237,239,241,243,245],[11], [27, 45, 47, 49, 187], [155, 159], [75], [],[151, 191, 193], [17, 21, 31, 33]]
    dvs=[]
    for index in dv_bna_index:
        if len(index)==0:
            dvs.append(0)
        else:
            dvs.append(np.mean(dv[np.asarray(index)]))
    return np.asarray(dvs)

def roi_brain_organization(atlas):
    if atlas=='gordon':
        latlas_file='%s/data/atlas/gordon/Parcels_L.func.gii'
        ratlas_file='%s/data/atlas/gordon/Parcels_R.func.gii'
        latlas=nib.load(latlas_file).agg_data()
        print(latlas)

def gordon_micrometrics():
    lsurf_164=nib.load('%s/data/atlas/gordon/Parcels_L_164.func.gii' % research_toolbox.__path__[0]).agg_data()
    rsurf_164=nib.load('%s/data/atlas/gordon/Parcels_R_164.func.gii' % research_toolbox.__path__[0]).agg_data()
    lsurf_32=nib.load('%s/data/atlas/gordon/Parcels_L.func.gii' % research_toolbox.__path__[0]).agg_data()
    rsurf_32=nib.load('%s/data/atlas/gordon/Parcels_R.func.gii' % research_toolbox.__path__[0]).agg_data()

    lhisgradient2=nib.load('%s/data/Brain_Organization/HistGradients_G2/HistGradients_G2.lh.fs32.func.gii' % research_toolbox.__path__[0]).agg_data()
    rhisgradient2=nib.load('%s/data/Brain_Organization/HistGradients_G2/HistGradients_G2.rh.fs32.func.gii' % research_toolbox.__path__[0]).agg_data()
    lhisgradient1=nib.load('%s/data/Brain_Organization/HistGradients_G1/HistGradients_G1.lh.fs32.func.gii' % research_toolbox.__path__[0]).agg_data()
    rhisgradient1=nib.load('%s/data/Brain_Organization/HistGradients_G1/HistGradients_G1.rh.fs32.func.gii' % research_toolbox.__path__[0]).agg_data()

    lmyelin=nib.load('%s/data/Brain_Organization/Myelin/source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-L_feature.func.gii' % research_toolbox.__path__[0]).agg_data()
    rmyelin=nib.load('%s/data/Brain_Organization/Myelin/source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-R_feature.func.gii' % research_toolbox.__path__[0]).agg_data()

    hisgradient1=np.zeros(333)
    hisgradient2=np.zeros(333)
    myelin=np.zeros(333)

    for i in range(1,162):
        hisgradient1[i-1]=np.nanmean(lhisgradient1[np.squeeze(lsurf_32)==i])
        hisgradient2[i-1]=np.nanmean(lhisgradient2[np.squeeze(lsurf_32)==i])
        myelin[i-1]=np.nanmean(lmyelin[np.squeeze(lsurf_32)==i])
    for i in range(162,334):
        hisgradient1[i-1]=np.nanmean(rhisgradient1[np.squeeze(rsurf_32)==i])
        hisgradient2[i-1]=np.nanmean(rhisgradient2[np.squeeze(rsurf_32)==i])
        myelin[i-1]=np.nanmean(rmyelin[np.squeeze(rsurf_32)==i])
    return [hisgradient2, myelin, hisgradient1]


def network_data(demo,nets,center_label,cognition):
    features=[]
    for net in nets:
        # features.append(net[np.triu_indices_from(net,k=1)])
        features.append(net)
    
    features=np.array(features)
    print(features.shape)
    features=features.astype(np.float32)
    infos=demo['center'].to_numpy().astype(np.float32)
    features=features[infos==center_label,:]
    labels=demo[cognition].to_numpy().astype(np.float32)
    labels=labels[infos==center_label]
    labels=labels[:,np.newaxis]
    if cognition=='exe':
        labels[labels<10]=np.nan
        labels[labels>300]=np.nan
    elif cognition=='memory':
        labels[labels<1]=np.nan
    elif cognition=='attention':
        labels[labels<10]=np.nan
    nan_index=np.isnan(labels)
    labels=labels[np.bitwise_not(nan_index[:,0]),:]
    return features, labels

def nested_2fold_cv_train(demo,nets,cognition,fold_n=2,is_act=True,norm_method='norm',repeat_num=10,test_flag=False,is_shuffle=False,is_rand=False):
    '''
    is_shuffle controls whether the data should be shuffled. It's used when using random 2 fold cross-validation.
    is_random controls whether to randomise the labels of training data. It's used when performing permutation test.
    '''
    hcp_features, hcp_labels=network_data(demo,nets,1,cognition)
    babri_features, babri_labels=network_data(demo,nets,3,cognition)
    ds=MyDataSets([hcp_features,babri_features],[hcp_labels,babri_labels],fold_n,3,is_shuffle)
    if not test_flag:
        l1_lambdas=[0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    else:
        l1_lambdas=[16]
    best_l1_lambdas=[]
    best_rvals=[]
    best_maes=[]
    best_models=[]
    best_trainers=[]
    best_trans=[]
    test_hats=[]
    test_labels=[]
    n=0
    # Outer CV for model evaluation
    for data in ds.nest_split(2,is_rand):
        print("/033[7;34m The %02d fold cross validation start /033[0m" % n)
        rvals=np.zeros((len(data['all_train_features']),len(l1_lambdas),repeat_num))
        maes=np.zeros((len(data['all_train_features']),len(l1_lambdas),repeat_num))
        # Inner CV for select the best hyper-parameters
        for k in range(len(data['all_train_features'])):
            for l in range(len(l1_lambdas)):
                l1_lambda=l1_lambdas[l]
                print("/033[7;34m L1 lambda is: %.5f /033[0m" % l1_lambda)
                # The resultant model is influenced by the initiate of parameters. So, we repeatively train 10 models, and use the median performance of the 10 models to evaluate the performance of a hyper-parameter.
                for i in range(repeat_num):
                    print("/033[7;34m For each inner CV, we run %d times and select the median performance to evaluate the performance of model. This is th %dth running /033[0m" % (repeat_num,i))
                    rval,model,trainer,tran,val_hat=train(data['all_train_features'][k],data['all_train_labels'][k],data['all_val_features'][k],data['all_val_labels'][k],l1_lambda,1, is_act, norm_method)
                    mae=np.mean(np.abs(val_hat[0].numpy().T-data['all_val_labels'][k].T))
                    print("/033[7;34m The rval is %.3f, mae is %.3f /033[0m" % (rval,mae))
                    rvals[k,l,i]=rval
                    maes[k,l,i]=mae
        maes=np.median(maes,axis=2)
        mean_acc=np.mean(maes,axis=0)
        best_l1_lambda=l1_lambdas[np.where(mean_acc==np.min(mean_acc))[0][0]]
        print("mean accuracy is ", mean_acc)
        print("best l1 lambda is %.5f" % best_l1_lambda)
        train_data=np.concatenate((data['all_train_features'][0],data['all_val_features'][0]),axis=0)
        train_label=np.concatenate((data['all_train_labels'][0],data['all_val_labels'][0]),axis=0)
        test_data=data['test_features']
        test_label=data['test_labels']

        rvals=[]; maes=[]; models=[]; trainers=[]; trans=[]; val_hats=[]; val_labels=[]
        for i in range(repeat_num):
            rval,model,trainer,tran,val_hat=train(train_data,train_label,test_data,test_label,best_l1_lambda,1,is_act, norm_method)
            mae=np.mean(np.abs(val_hat[0].numpy().T-test_label.T))
            maes.append(mae)
            models.append(model)
            trainers.append(trainer)
            trans.append(tran)
            rvals.append(rval)
            val_hats.append(val_hat[0].numpy().T[0])
            val_labels.append(test_label.T[0])
            print("/033[7;34m The r value is %.5f and mae is %.2f /033[0m" % (rval,mae))
        print(rvals)
        print(maes)
        rvals=np.asarray(rvals)
        maes=np.asarray(maes)

        mean_acc=maes
        median_acc=np.sort(maes)[int(repeat_num/2)-1]
        best_l1_lambdas.append(best_l1_lambda)
        best_rvals.append(rvals[mean_acc==median_acc])
        best_maes.append(maes[mean_acc==median_acc])
        print(np.where(mean_acc==median_acc)[0][0])
        best_models.append(models[np.where(mean_acc==median_acc)[0][0]])
        best_trainers.append(trainers[np.where(mean_acc==median_acc)[0][0]])
        best_trans.append(trans[np.where(mean_acc==median_acc)[0][0]])
        test_hats.append(val_hats[np.where(mean_acc==median_acc)[0][0]])
        test_labels.append(val_labels[np.where(mean_acc==median_acc)[0][0]])
        
        print("/033[7;34m The best l1_lambdas are /033[0m", best_l1_lambdas)
        print("/033[7;34m The best rvals are /033[0m", best_rvals)
        print("/033[7;34m The best maes are /033[0m", best_maes)
        n+=1

    return best_l1_lambdas, best_rvals, best_maes, best_models, best_trainers, best_trans, test_hats, test_labels

if __name__=='__main__':
    print(roi_brain_organization('gordon'))