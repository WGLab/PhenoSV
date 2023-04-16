import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import utilities.utility as u
import model.operation_function as of
import argparse
import pandas as pd
import Phen2Gene.phen2gene as pg
import model.data_loader as dat
import torch.utils.data as data_utils
import numpy as np
import torch
from joblib import Parallel, delayed


parser = argparse.ArgumentParser(description='SV Prioritization')

#input sv
parser.add_argument('--patho_sv_path',type=str,default='/home/xu3/Project_PhenoSV/data/SV/meta/svanna_coding_pheno.csv')

#input background file
parser.add_argument('--background_sv_path',type=str,default='/home/xu3/Project_PhenoSV/data/SV/meta/background_data.bed')

#set up file paths
parser.add_argument('--feature_folder',type=str)#'/scr1/users/xu3/phenosv/sv_feature/svfeatures/'--set up this file to used pre-extracted features for background
parser.add_argument('--ckpt_folder', type=str, default='/scr1/users/xu3/phenosv/results/2022-12-19 08:19:28.359510')
parser.add_argument('--elements_path',type=str,default='/scr1/users/xu3/phenosv/features/genes_w_noncoding.bb')
parser.add_argument('--feature_files', type=str, default='/home/xu3/Project_PhenoSV/data/Features/featuremaster_chop1026.csv')
parser.add_argument('--scaler_file',type=str,default='/home/xu3/Project_PhenoSV/data/Features/features1026.csv')
parser.add_argument('--annotation_path', type=str, default='/scr1/users/xu3/phenosv/features/exon_gencode.bed')
parser.add_argument('--tad_path', type=str)
#input phen2gene
parser.add_argument('--KBpath',type=str,default='/home/xu3/Phen2Gene/lib')
parser.add_argument('--alpha',type=float,default=1)#HPO speicificity, positive, when value>1, add contrast of phen2gene scores; value<1 decrease contrast

#simulation setting
parser.add_argument('--n', type=int, default=20000)
parser.add_argument('--n_workers', type=int, default=1)



def perform_prioritizatin(input_sv, model, i, target_folder, drop_common):
    #simulate background
    background_sv = u.simulation_background(args.background_sv_path, n=args.n, drop_common=drop_common)
    sv = input_sv.loc[i:i, ['CHR', 'START', 'END', 'SVTYPE', 'ID']]
    sv = pd.concat([sv, background_sv]).reset_index(drop=True)
    sv['PATHO'] = [1] + [0] * background_sv.shape[0]
    ID = sv['ID'][0]
    print(ID)
    # pheno---gene
    gene_list = pg.phen2gene(KBpath=args.KBpath, HPO=input_sv['HPO'][i], scale_score=True)
    sv['PATH'] = [os.path.join(args.feature_folder, id + '.npz') if os.path.isfile(os.path.join(args.feature_folder, id + '.npz')) else None for id in sv['ID'].tolist()]
    sv_presaved = sv[sv['PATH'].isnull()==False].reset_index(drop=True)
    sv_onthefly = sv[sv['PATH'].isnull() == True].reset_index(drop=True)
    pred_presaved = perform_prioritization_presaved(sv_presaved, gene_list, model)
    pred_onthefly = perform_prioritization_onethefly(sv_onthefly, gene_list, model)
    pred = pd.concat([pred_presaved,pred_onthefly]).reset_index(drop=True)
    pred.to_csv(os.path.join(target_folder, ID + '.csv'))

def perform_prioritization_presaved(sv_df, gene_list, model):
    #use pre-saved features for prioritizations, this can save time of generating features of the same SVs repeatly

    dset_test = dat.SVLocalDataset(sv_df, y_col='PATHO', feature_subset=None)
    dat_loader = data_utils.DataLoader(dset_test, batch_size=1, shuffle=False,
                                       collate_fn=dat.collate_fn_padd_local)
    df = []
    for batch_idx, batch in enumerate(dat_loader):
        x, mask, gene_indicators, sv_indicators, y = batch
        sv_pred, _, element_pred = model.get_element_score(batch)
        file = np.load(sv_df['PATH'][batch_idx])
        id = [sv_df['ID'][batch_idx]]
        element_names = file['element_names'].tolist()
        if len(element_pred) > len(element_names):
            element_pred = element_pred[1:-1]
        # coding
        if torch.sum(1 - sv_indicators) == 0:
            element_names = ['SV'] + element_names
            type_indicator = ['Exonic'] * len(element_names)
            type_indicator[0] = 'Coding SV'
            sv_pred = [sv_pred] + element_pred
            sv_pred_cali = of.calibrate_confidence(sv_pred, cutoff=0.4934)
            # get results
            df_ = pd.DataFrame({'Elements': element_names, 'Pathogenecity': sv_pred_cali, 'Type': type_indicator})
            df_ = df_[df_.Elements!='noncoding'].reset_index(drop=True)
            df_['ID'] = id * df_.shape[0]
        else:  # noncoding
            CHR = sv_df.loc[batch_idx, 'CHR']
            START = sv_df.loc[batch_idx, 'START']
            END = sv_df.loc[batch_idx, 'END']
            gene_names, intronic_gene_names = of.get_global_genes(CHR, START, END, elements_path=args.elements_path,
                                                                  tad_path=args.tad_path)
            element_names = [e.replace('_sv', '') if '_sv' in e else e for e in element_names]
            df_ = pd.DataFrame({'element_names': element_names, 'element_pred': element_pred})
            df_ = df_[df_.element_names != 'noncoding'].reset_index(drop=True)
            df_ = df_.groupby('element_names').agg('max')
            element_names = ['SV'] + df_.index.tolist()
            type_indicator = ['Intronic' if g in intronic_gene_names else 'Regulatory' for g in element_names]
            type_indicator[0] = 'Non-coding SV'
            sv_pred = [sv_pred] + df_.element_pred.tolist()
            sv_pred_cali = of.calibrate_confidence(sv_pred, cutoff=0.7901)
            df_ = pd.DataFrame(
                {'Elements': element_names, 'Pathogenecity': sv_pred_cali, 'Type': type_indicator})
            df_['ID'] = id * df_.shape[0]
        df.append(df_)
    pred = pd.concat(df)
    if gene_list is not None:
        pred = pred.merge(gene_list[['Gene', 'Score']], how='left', left_on='Elements', right_on='Gene')
        pred['PhenoSV'] = pred['Pathogenecity'] * (pred['Score'] ** args.alpha)
    return pred


def perform_prioritization_presaved_othertype(sv_df, gene_list, model):
    return

def perform_prioritization_onethefly(sv_df, gene_list, model):
    # prioritization on the fly
    pred = of.phenosv(None, None, None, None, sv_df, args.annotation_path, model, args.elements_path,
                           args.feature_files, args.scaler_file, args.tad_path, cutoff_coding=0.4934,
                           cutoff_noncoding=0.7901,HPO=None, pheno_adjust=args.alpha)
    if gene_list is not None:
        pred = pred.merge(gene_list[['Gene', 'Score']], how='left', left_on='Elements', right_on='Gene')
        pred['PhenoSV'] = pred['Pathogenecity'] * (pred['Score'] ** args.alpha)
    return pred

def main():
    global args
    args = parser.parse_args()
    ckpt_path= [os.path.join(args.ckpt_folder, i) for i in os.listdir(args.ckpt_folder) if 'ckpt' in i][0]
    #prepare models
    print('loading model')
    model = of.prepare_model(ckpt_path)

    print('reading pathogenic sv and background')
    if 'csv' in args.patho_sv_path:
        input_sv = pd.read_csv(args.patho_sv_path)
    else:
        input_sv = pd.read_csv(args.patho_sv_path,sep='\t')

    print('creating target folder')
    target_folder = os.path.join(args.ckpt_folder, "prioritization")
    if os.path.isdir(target_folder) == False:
        os.makedirs(target_folder)
        
    #print('skip existing SVs')
    #existing = os.listdir(target_folder)
    #if len(existing)>0:
    #    ind = [i for i, f in enumerate(input_sv.ID.tolist()) if f + '.csv' not in existing]
    #    input_sv = input_sv.iloc[ind].reset_index(drop=True)

    print('start simulation')
    if args.n_workers==1:
        for i in range(input_sv.shape[0]):
            print(input_sv['ID'][i])
            perform_prioritizatin(input_sv, model, i, target_folder, drop_common=True)
    else:
        print('parallel computing')
        Parallel(n_jobs=args.n_workers)(perform_prioritizatin(input_sv, model, i, target_folder, drop_common=True) for i in range(input_sv.shape[0]))


if __name__ == '__main__':
    main()