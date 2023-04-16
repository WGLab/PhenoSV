import sys
import os
import argparse
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import utilities.utility as u
import utilities.read_features as rf
import pandas as pd
from model.phenosv import init as init


parser = argparse.ArgumentParser(description = 'Generate SV Feature Matrix for deep learning model')
#mandatory
parser.add_argument('--sv_file', type=str, help="path of SV file， accept csv and bed formats")
parser.add_argument('--target', type = str, default='.', help="path to save feature files")
parser.add_argument('--noncoding',type=str,default='distance',help='inference mode, choose from distance and tad')

#optional
parser.add_argument('--skip', type=int)#set to 1 if use

def main():
    global args
    args = parser.parse_args()
    if os.path.isdir(args.target)==False:
        os.makedirs(args.target)

    #init
    configs = init()
    feature_files, scaler_file, _, elements_path, annotation_path, tad_path, _ = list(configs.values())
    if args.noncoding=='distance':
        tad_path = None

    #read sv files
    sv_df = None
    if args.sv_file is not None:
        assert args.sv_file.endswith('.bed') or args.sv_file.endswith('.csv'), f"please input a bed file or a csv file"
        if args.sv_file.endswith('.bed'):
            _, _, sv_df = u.read_bed(args.sv_file, N=None, parse=False)
            if sv_df.shape[1] == 6:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE', 'HPO']
            else:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE']
        else: #args.sv_file.endswith('.csv'):
            sv_df = pd.read_csv(args.sv_file, header=None)
            if sv_df.shape[1] == 6:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE', 'HPO']
            else:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE']

    valid_chromosomes = ['chr' + str(i) for i in range(1, 23)]
    sv_df = sv_df[sv_df.iloc[:, 0].isin(valid_chromosomes)].reset_index(drop=True)
    # skip files already exist
    if args.skip==1:
        files = os.listdir(args.target)
        if len(files)>0:
            ind = [i for i, f in enumerate(list(sv_df.iloc[:, 3])) if f + '.npz' not in files]
            sv_df = sv_df.iloc[ind].reset_index(drop=True)
            ind = [i for i, f in enumerate(list(sv_df.iloc[:, 3])) if f not in files]
            sv_df = sv_df.iloc[ind].reset_index(drop=True)
            print('extracting features for '+str(sv_df.shape[0])+' SVs')


    sv_df = sv_df[sv_df.iloc[:, 4].isin(['deletion', 'duplication','inversion','insertion'])]
    assert sv_df.shape[0]>0, f"please input a valid SV file"
    rf.save_sv_features_auto(sv_df, args.target, feature_files, annotation_path, elements_path, tad_path, scaler_file)

if __name__ == '__main__':
    main()




