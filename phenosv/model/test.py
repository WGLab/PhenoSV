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
import model.operation_function as of

parser = argparse.ArgumentParser(description='SV TRANSFORMER MODEL')
#------overall setting
parser.add_argument('--test_type',type=str,default='overall')#"overall","element","model"
parser.add_argument('--model_dir', type=str, default='/athena/marchionnilab/scratch/zhx2006/results/2022-07-19 21:28:11/')
parser.add_argument('--model_type', type=str, default='local')#choose from local or sv_gene
parser.add_argument('--df_path_test', type=str, default='/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/test_clinvar.csv')
parser.add_argument('--feature_path', type=str, default='/athena/marchionnilab/scratch/zhx2006/sv_feature/local/')
parser.add_argument('--test_name', type=str, default='clinvar_test')
#features-wise
parser.add_argument('--features', type=str, default='all')
parser.add_argument('--feature_group',type=str)
parser.add_argument('--feature_grouping_path', type=str)#leave it empty if not grouping

#------test prioritization setting
parser.add_argument('--benign_df_path', type=str, default='/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/sv_lr_rare_coding.csv')
parser.add_argument('--patho_df_path', type=str,default='/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/test_decipher_coding_prioritization.csv')
parser.add_argument('--alpha', type=float, default=3)
parser.add_argument('--workers', type=int, default=8)

#------test model setting--including coding and noncoding
parser.add_argument('--elements_path', type=str, default='/scr1/users/xu3/phenosv/features/genes_w_noncoding.bb')
parser.add_argument('--feature_files', type=str, default='/home/xu3/Project_PhenoSV/data/Features/featuremaster_chop1026.csv')
parser.add_argument('--scaler_file', type=str, default='/home/xu3/Project_PhenoSV/data/Features/features1026.csv')
parser.add_argument('--tad_path', type=str) #default use None
parser.add_argument('--annotation_path', type=str, default='/scr1/users/xu3/phenosv/features/exon_gencode.bed')
parser.add_argument('--partial_mode', action='store_true')
parser.add_argument('--full_mode', action='store_false',dest='partial_mode')#set to full_mode to consider both direct and indirect effects of coding SVs



def get_prediction(ckpt_folder,test_path, feature_path, test_name, feature_subset=None):
    #model ckpt path
    ckpt_path = [os.path.join(ckpt_folder, i) for i in os.listdir(ckpt_folder) if 'ckpt' in i][0]

    #df of test data to predict
    test_df = pd.read_csv(test_path)
    test_df['PATH']= [os.path.join(feature_path ,i+'.npz') for i in test_df['ID'].tolist()]

    #set up dataloader and model
    model = mm.LocalSVModule.load_from_checkpoint(ckpt_path)
    DATA = dat.DataModule(test_df=test_df, feature_subset=feature_subset)

    trainer = pl.Trainer()

    #predict
    predictions = trainer.predict(model, DATA)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.columns = ['prediction', 'truth']
    prediction_df['ID'] = list(test_df.ID)

    #get matrics
    fpr, tpr, _ = metrics.roc_curve(np.array(prediction_df.truth),np.array(prediction_df.prediction), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(np.array(prediction_df.truth),np.round(np.array(prediction_df.prediction),0).astype(int))
    tn, fp, fn, tp =metrics.confusion_matrix(np.array(prediction_df.truth), np.round(np.array(prediction_df.prediction),0).astype(int)).ravel()
    sen=tp/(tp+fn)
    spe=tn/(tn+fp)
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = metrics.f1_score(np.array(prediction_df.truth), np.round(np.array(prediction_df.prediction), 0).astype(int))
    pcroc = metrics.average_precision_score(np.array(prediction_df.truth), np.array(prediction_df.prediction))
    performance_df = pd.DataFrame({'auc':[auc], 'acc':[acc], 'sen': [sen], 'spe': [spe],
                                   'precision':[pre],'recall':[rec],'f1_score':[f1], 'pcroc':pcroc})

    #save
    if test_name is None:
        test_name='test'
    prediction_df.to_csv(os.path.join(ckpt_folder,str(test_name)+'_predictions.csv'))
    performance_df.to_csv(os.path.join(ckpt_folder,str(test_name)+'_prediction_performance.csv'))



def main():
    global args
    args = parser.parse_args()
    if args.features == 'all':
        feature_group_index = None
    else:
        df = pd.read_csv(args.feature_grouping_path)
        if ',' in args.feature_group:
            feature_group = [int(i) for i in args.feature_group.split(',')]
        else:
            feature_group=[int(args.feature_group)]
        feature_group_index = df[df['group'].isin(feature_group)]['index'].tolist()
        feature_group_index.append(-1)

    if args.test_type=='overall':
        get_prediction(ckpt_folder=args.model_dir, test_path=args.df_path_test, feature_path=args.feature_path,
                       test_name=args.test_name, feature_subset=feature_group_index)
    elif args.test_type=='element':
        dest_file = os.path.join(args.model_dir, str(args.test_name) + '_element_predictions.csv')
        of.element_scores_local(ckpt_folder=args.model_dir,test_path=args.df_path_test,feature_path=args.feature_path,
                                    dest_file=dest_file, feature_subset=feature_group_index)
    else: #args.test_type == 'model':
        dest_file = os.path.join(args.model_dir, str(args.test_name) + '_modelpredictions.csv')
        sv_df = pd.read_csv(args.df_path_test)
        ckpt_path = [os.path.join(args.model_dir, i) for i in os.listdir(args.model_dir) if 'ckpt' in i][0]
        model = of.prepare_model(ckpt_path)
        if 'HPO' in list(sv_df.columns):
            HPO = sv_df['HPO']
        else:
            HPO=None

        if args.partial_mode:
            full_mode = False
        else:
            full_mode=True
        pred = of.phenosv(CHR=None, START=None, END=None, svtype=None,sv_df=sv_df, annotation_path=args.annotation_path, model=model,
                          elements_path=args.elements_path,feature_files=args.feature_files, scaler_file=args.scaler_file,tad_path = args.tad_path,
                            cutoff_coding=0.4934, cutoff_noncoding=0.7901, HPO=HPO, pheno_adjust=1, full_mode = full_mode,feature_subset=feature_group_index)
        pred.to_csv(dest_file)

if __name__ == '__main__':
    main()