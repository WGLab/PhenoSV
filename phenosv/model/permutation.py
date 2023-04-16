import os
import sys
import argparse
import pandas as pd
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import model.model as mm
import model.data_loader as dat
import pytorch_lightning as pl
from sklearn import metrics
import numpy as np


parser = argparse.ArgumentParser(description='SV TRANSFORMER MODEL')
#dataset
parser.add_argument('--model_dir', type=str, default='/athena/marchionnilab/scratch/zhx2006/results/2022-07-19 21:28:11/')
parser.add_argument('--df_path_test', type=str, default='../../data/SV/meta/test_clinvar.csv')
parser.add_argument('--feature_path', type=str, default='/athena/marchionnilab/scratch/zhx2006/sv_feature/local/')
parser.add_argument('--test_name', type=str, default='clinvar_test')
parser.add_argument('--ref_mat_path', type=str, default='/scr1/users/xu3/phenosv/features/ref_features_matrix.npy')
parser.add_argument('--perm_time',type=int,default=1)



def get_prediction_permutation(ckpt_folder,test_path,feature_path, perm,ref_mat_path,feature_subset=None):
    #model ckpt path
    ckpt_path = [os.path.join(ckpt_folder, i) for i in os.listdir(ckpt_folder) if 'ckpt' in i][0]

    #df of test data to predict
    test_df = pd.read_csv(test_path)
    test_df['PATH']= [os.path.join(feature_path ,i+'.npz') for i in test_df['ID'].tolist()]

    #set up dataloader and model

    model = mm.LocalSVModule.load_from_checkpoint(ckpt_path)
    DATA = dat.DataModule(test_df=test_df,  feature_subset=feature_subset, perm=perm, ref_mat_path=ref_mat_path)

    trainer = pl.Trainer()

    #predict
    predictions = trainer.predict(model, DATA)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.columns = ['prediction', 'truth']
    prediction_df['ID'] = list(test_df.ID)

    #get matrics
    fpr, tpr, _ = metrics.roc_curve(np.array(prediction_df.truth),
                                    np.array(prediction_df.prediction), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(np.array(prediction_df.truth),np.round(np.array(prediction_df.prediction),0).astype(int))

    return acc, auc

def perform_permutation(ckpt_folder,test_path,feature_path,input_dim, perm_time,test_name, ref_mat_path,feature_subset=None):
    folder = os.path.join(ckpt_folder, 'feature_importance')
    if os.path.exists(folder)==False:
        os.mkdir(folder)
    #baseline
    acc0, auc0 = get_prediction_permutation(ckpt_folder, test_path, feature_path, perm=None, ref_mat_path=ref_mat_path,feature_subset=feature_subset)
    #permutation
    acc=[acc0]
    auc=[auc0]
    for p in range(input_dim):
        acc_p=[]
        auc_p=[]
        for t in range(perm_time):
            acc_, auc_ = get_prediction_permutation(ckpt_folder, test_path, feature_path, perm=p, ref_mat_path=ref_mat_path,feature_subset=feature_subset)
            acc_p.append(acc_)
            auc_p.append(auc_)
        acc_p = np.mean(acc_p)
        auc_p = np.mean(auc_p)
        acc.append(acc_p)
        auc.append(auc_p)
    df=pd.DataFrame({'acc':acc, 'auc':auc})

    df.to_csv(os.path.join(folder, test_name+'_feature_permutations.csv'))



def main():
    global args
    args = parser.parse_args()
    perform_permutation(ckpt_folder=args.model_dir, test_path=args.df_path_test, feature_path=args.feature_path,
                        input_dim=238, perm_time=args.perm_time,test_name=args.test_name,ref_mat_path=args.ref_mat_path,
                        feature_subset=None)
if __name__ == '__main__':
    main()








