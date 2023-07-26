import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import utilities.utility as u
import model.operation_function as of
import argparse
import pandas as pd
from liftover import get_lifter


parser = argparse.ArgumentParser(description='PhenoSV: phenotype-aware structural variants scoring')
parser.add_argument('--config',action='store_true')
parser.add_argument('--config_omit',dest='config', action='store_false')
#settings
parser.add_argument('--genome', type=str, default='hg38', help="choose genome build between hg38 (default) and hg19")
parser.add_argument('--alpha',type=float,default=1.0, required=False, help="A positive value with larger value representing more contribution of phenotype information in refining PhenoSV scores. Default is 1")
parser.add_argument('--inference', type=str, help="leave it blank (default) if only considering direct impacts of coding SVs. Set to `full` if inferring both direct and indirect impacts of coding SVs")
parser.add_argument('--model', type=str, default='PhenoSV', help="choose between PhenoSV (default) and PhenoSV-light")

#input single sv
parser.add_argument('--c',type=str, help='chromosome, e.g. chr1')#chrom
parser.add_argument('--s',type=int, help='start, e.g. 2378909')#start
parser.add_argument('--e',type=int, help='end, e.g. 2379909')#end
parser.add_argument('--c2',type=str, help='chromosome2, e.g. chr1, only for translocation')
parser.add_argument('--s2',type=int, help='start2, e.g. 2378909, only for translocation')
parser.add_argument('--strand1',type=str, default='+',help='strand1, + or - , only for translocation')
parser.add_argument('--strand2',type=str, default='+',help='strand2, + or - , only for translocation')
parser.add_argument('--svtype',type=str, help='deletion, duplication, insertion, inversion, translocation')
parser.add_argument('--noncoding',type=str,default='distance',help='inference mode, choose from distance and tad')
parser.add_argument('--HPO',type=str,required=False, help='HPO terms should in the format of HP:digits, e.g., HP:0000707, separated by semicolons, commas, or spaces.')


#input multiple SVs in bed/csv format
parser.add_argument('--sv_file',type=str,help='path to SV file (.csv, .bed, .bedpe)')

#output setting
parser.add_argument('--target_folder',type=str,help='enter the folder path to save PhenoSV results, leave it blank if you only want to print out the results')
parser.add_argument('--target_file_name',type=str,default='sv_score',help='enter the file name to save PhenoSV results')




def init(configpath=None, ckpt=False, light = False):
    if configpath is None:
        configpath = os.path.join(module_dir,'../lib/fpath.config')
    with open(configpath) as fr:
        KBPATH = fr.readline().rstrip('\n')
    Path = os.path.join(KBPATH,'data')
    if light:
        feature_files=os.path.join(Path,'features_set_light.csv')
        scaler_file = os.path.join(Path, 'features1026_light.csv')
        ckpt_path = os.path.join(Path, 'model-epoch=57-val_loss=0.34.ckpt')
    else:
        feature_files=os.path.join(Path,'features_set.csv')
        scaler_file = os.path.join(Path, 'features1026.csv')
        ckpt_path = os.path.join(Path, 'model-epoch=33-val_loss=0.33.ckpt')

    elements_path = os.path.join(Path, 'genes_w_noncoding.bb')
    annotation_path = os.path.join(Path, 'exon_gencode.bed')
    tad_path = os.path.join(Path, 'tad_w_boundary_08.bed')
    if ckpt:
        return {'feature_files': feature_files, 'scaler_file': scaler_file, 'elements_path': elements_path,
                'annotation_path': annotation_path, 'tad_path': tad_path, 'KBpath': KBPATH}, ckpt_path,
    else:
        return {'feature_files':feature_files,'scaler_file':scaler_file,'ckpt_path':ckpt_path,'elements_path':elements_path,
            'annotation_path':annotation_path,'tad_path':tad_path,'KBpath':KBPATH}


def main():
    global args
    args = parser.parse_args()
    configs = init()
    feature_files, scaler_file, ckpt_path, elements_path, annotation_path, tad_path, KBPATH = list(configs.values())
    if args.config:
        print(configs)
        exit()

    #genome build
    assert args.genome=='hg19' or args.genome=='hg38',f' only support hg19 and hg38'
    if args.genome=='hg19':
        converter = get_lifter('hg19', 'hg38')
    else:
        converter = None

    #prepare models
    model = of.prepare_model(ckpt_path)
    if args.inference=='full':
        full_mode = True
    else:
        full_mode = False
    if args.noncoding == 'distance':
        tad_path = None
    if args.model=='PhenoSV-light':
        feature_subset = [0, 2, 12, 13, 14, 16, 21, 22, 23, 25, 26, 27, 30, 34, 35, 37, 40, 42, 43, 44,
                         48, 50, 51, 54, 57, 61, 64, 70, 71, 80, 86, 94, 119, 136, 165, 175, 178, 184, 189, 223, 224,
                         -1]
    else:
        feature_subset = None

    if args.sv_file is not None:
        assert args.sv_file.endswith('.bed') or args.sv_file.endswith('.csv') or args.sv_file.endswith('.bedpe'), f"please input a bed file or a csv file"
        if args.sv_file.endswith('.bed'):
            _, _, sv_df = u.read_bed(args.sv_file, N=None, parse=False)
            if sv_df.shape[1] == 6:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE', 'HPO']
            else:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE']
        elif args.sv_file.endswith('.csv'):
            sv_df = pd.read_csv(args.sv_file, header=None)
            if sv_df.shape[1] == 6:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE', 'HPO']
            else:
                sv_df.columns = ['CHR', 'START', 'END', 'ID', 'SVTYPE']
        else:
            sv_df = pd.read_csv(args.sv_file,sep='\t', header=None)
            if sv_df.shape[1]==10:
                sv_df.columns = ['CHR1','START1','END1','CHR2','START2','END2', 'STRAND1','STRAND2','ID','HPO']
            else:
                sv_df.columns = ['CHR1','START1','END1','CHR2','START2','END2', 'STRAND1','STRAND2','ID']

        valid_chromosomes = ['chr' + str(i) for i in range(1, 23)]
        sv_df = sv_df[sv_df.iloc[:,0].isin(valid_chromosomes)].reset_index(drop=True)

        if 'HPO' in list(sv_df.columns):
            HPO = sv_df['HPO']
        elif args.HPO is not None:
            HPO = args.HPO
        else:
            HPO=None

        if converter is not None:
            if 'START' in list(sv_df.columns):
                sv_df['START'] = [u.liftover(sv_df['CHR'][0], start, converter) for start in sv_df['START'].tolist()]
                sv_df['END'] = [u.liftover(sv_df['CHR'][0], start, converter) for start in sv_df['END'].tolist()]
                sv_df = sv_df.drop(sv_df[(sv_df['START'] == -1) | (sv_df['END'] == -1)].index).reset_index(drop=True)
            if 'START1' in list(sv_df.columns):
                sv_df['START1'] = [u.liftover(sv_df['CHR1'][0], start, converter) for start in sv_df['START1'].tolist()]
                sv_df['START2'] = [u.liftover(sv_df['CHR2'][0], start, converter) for start in sv_df['START2'].tolist()]
                sv_df = sv_df.drop(sv_df[(sv_df['START1'] == -1) | (sv_df['START2'] == -1)].index).reset_index(drop=True)

        pred = of.phenosv(None, None, None, None, sv_df, annotation_path, model, elements_path, feature_files, scaler_file,
                   tad_path,cutoff_coding=0.4934, cutoff_noncoding=0.7901, HPO=HPO, pheno_adjust=args.alpha,
                   KBpath=KBPATH,full_mode=full_mode, CHR2=None, START2=None, strand1='+', strand2='+', feature_subset=feature_subset)
    else: #single SV
        s, e, s2 = args.s, args.e, args.s2
        if converter is not None:
            s = u.liftover(args.c, args.s, converter=converter)
            e = u.liftover(args.c, args.e, converter=converter)
            if args.s2 is not None:
                s2 = u.liftover(args.c, args.s2, converter=converter)
        if args.svtype=='translocation':
            assert args.c2 is not None and args.s2 is not None, f'for translocations, input c2 and s2 are required'
        if args.svtype=='insertion' and args.e is None:
            e = s+1

        pred=of.phenosv(args.c, s, e, args.svtype, None, annotation_path, model, elements_path, feature_files, scaler_file,
                   tad_path, cutoff_coding=0.4934, cutoff_noncoding=0.7901, HPO=args.HPO, pheno_adjust=args.alpha,
                   KBpath=KBPATH, full_mode=full_mode, CHR2=args.c2, START2=s2, strand1=args.strand1, strand2=args.strand2,
                        feature_subset=feature_subset)


    if args.target_folder is None:
        print(pred)
    else:
        pred.to_csv(os.path.join(args.target_folder,str(args.target_file_name)+'.csv'))

if __name__ == '__main__':
    main()
