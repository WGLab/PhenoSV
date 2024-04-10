
import os
import sys
import argparse
import pandas as pd
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import model.operation_function as of



parser = argparse.ArgumentParser(description='SV TRANSFORMER MODEL')
#generate attention map based on existing feature file
parser.add_argument('--ckpt_path', type=str, default='/scr1/users/xu3/phenosv/results/2022-12-19 08:19:28.359510/model-epoch=33-val_loss=0.33.ckpt')
parser.add_argument('--target_folder',type=str,default='/scr1/users/xu3/phenosv/results/2022-12-19 08:19:28.359510/interpretation/')
parser.add_argument('--elements_path', type=str, default='/scr1/users/xu3/phenosv/features/genes_w_noncoding.bb')
parser.add_argument('--feature_files', type=str, default='/home/xu3/Project_PhenoSV/data/Features/featuremaster_chop1026.csv')
parser.add_argument('--scaler_file', type=str, default='/home/xu3/Project_PhenoSV/data/Features/features1026.csv')
parser.add_argument('--annotation_path', type=str, default='/scr1/users/xu3/phenosv/features/exon_gencode.bed')
parser.add_argument('--tad_path', type=str)#'/scr1/users/xu3/phenosv/features/tad_w_boundary_08.bed'

#optional--either use dataframe or type sv location mannually
parser.add_argument('--sv_id', type=str)
parser.add_argument('--sv_df_path', type=str)
parser.add_argument('--CHR', type=str)
parser.add_argument('--START', type=int)
parser.add_argument('--END', type=int)
parser.add_argument('--SVTYPE', type=str)
parser.add_argument('--cache', type=str, default=None, help='path to data cache')
parser.add_argument('--force_noncoding', action='store_true')
parser.add_argument('--force_noncoding_off', action='store_false',dest='force_noncoding')


def main():
    global args
    args = parser.parse_args()
    if args.sv_df_path is not None:
        df = pd.read_csv(args.sv_df_path)
        assert args.sv_id is not None, "sv_id cannot be None when set to read sv_df_path"
        CHR = df['CHR'][df['ID']==args.sv_id].tolist()[0]
        START = df['START'][df['ID'] == args.sv_id].tolist()[0]
        END = df['END'][df['ID'] == args.sv_id].tolist()[0]
        SVTYPE = df['SVTYPE'][df['ID'] == args.sv_id].tolist()[0]
    else:
        CHR = args.CHR
        START = args.START
        END = args.END
        SVTYPE = args.SVTYPE
    if os.path.isdir(args.target_folder) is False:
        os.makedirs(args.target_folder)
    of.input_gradient(CHR, START, END, SVTYPE, args.elements_path, args.feature_files, args.scaler_file, args.ckpt_path,
                      annotation_path=args.annotation_path, cache=args.cache, target_file_name=args.sv_id,target_folder=args.target_folder,
                      tad_path=args.tad_path, force_noncoding = args.force_noncoding)
    of.attention_map(CHR, START, END, SVTYPE, args.elements_path, args.feature_files, args.scaler_file, args.ckpt_path,
                      annotation_path=args.annotation_path, cache=args.cache, target_file_name=args.sv_id,target_folder=args.target_folder,
                      tad_path=args.tad_path, force_noncoding=args.force_noncoding)
    print('finished')




if __name__ == '__main__':
    main()


