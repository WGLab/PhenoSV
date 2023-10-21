
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.optim as optim
import pandas as pd
import model.model as mm
import model.data_loader as dat
import torch.utils.data as data_utils
import torch
import pyBigWig
import utilities.read_features as rf
import utilities.utility as u
from pybedtools import BedTool
import Phen2Gene.phen2gene as pg


#model-based functions
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
def generate_attention_mask(gene_indicators, sv_indicators, num_heads):
    ##----head type 1
    # genes outside sv attend all sv; gene inside sv attend all within noncoding
    # 12/19 count padding as sv
    #sv_indicators_pad = sv_indicators.clone()
    sv_indicators[..., 0] = 1
    sv_indicators[..., -1] = 1
    gene_out_sv = gene_indicators * (1 - sv_indicators)
    gene_in_sv = gene_indicators * sv_indicators
    gout_sv = torch.bmm(torch.unsqueeze(gene_out_sv, 2).float(), torch.unsqueeze(sv_indicators, 1).float())
    ##
    gin_svn = torch.bmm(torch.unsqueeze(gene_in_sv, 2).float(),
                        torch.unsqueeze((1 - gene_indicators) * sv_indicators, 1).float())
    # all noncoding attend themselves
    n_n = torch.bmm(torch.unsqueeze(1 - gene_indicators, 2).float(), torch.unsqueeze(1 - gene_indicators, 1).float())
    diag_mask = torch.eye(n_n.size()[1]).repeat(n_n.size()[0], 1, 1).bool()
    n_n[diag_mask == False] = 0
    # sum
    indicators_mat1 = gout_sv + gin_svn + n_n

    ##----head type2:  all attend themselves
    indicators_mat2 = n_n
    indicators_mat2[diag_mask == False] = 0
    indicators_mat2[diag_mask == True] = 1

    indicators_mat1 = indicators_mat1.unsqueeze(1).repeat(1, num_heads // 2, 1, 1)  # batch head seq seq
    indicators_mat2 = indicators_mat2.unsqueeze(1).repeat(1, num_heads // 2, 1, 1)  # batch head seq seq
    indicators_mat = torch.cat((indicators_mat1, indicators_mat2), 1)
    return indicators_mat


def soft_cce_loss(y_logit, y_true, reduction='mean',weighted = False):
    y_pred = torch.sigmoid(y_logit)
    y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
    if weighted:
        w=torch.sum(y_true==0)/torch.clamp(torch.sum(y_true>0),min=1e-6).detach()#pos weight
        loss = -(w*y_true * torch.log(y_pred) + (1 - y_true) * torch.log((1 - y_pred)))
    else:
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log((1 - y_pred)))
    if reduction=='mean':
        loss = torch.sum(loss) / loss.shape[0]
    return loss

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


#batch size must be one
def element_scores_local(ckpt_folder,test_path,feature_path,dest_file, feature_subset=None):
    '''
    ckpt_folder: folder path for ckpt
    test_path: csv file path for test SVs: '/athena/marchionnilab/scratch/zhx2006/codes/Project_PhenoSV/data/SV/meta/test_clinvar.csv'
    feature_path: sv_feature folder path: '/athena/marchionnilab/scratch/zhx2006/sv_feature/local'
    dest_file: save file as
    '''
    ckpt_path = [os.path.join(ckpt_folder, i) for i in os.listdir(ckpt_folder) if 'ckpt' in i][0]
    model = mm.LocalSVModule.load_from_checkpoint(ckpt_path).eval()

    # df of test data to predict
    test_df = pd.read_csv(test_path)
    test_df['PATH'] = [os.path.join(feature_path, i + '.npz') for i in test_df['ID'].tolist()]

    dset_test = dat.SVLocalDataset(test_df, y_col='PATHO', feature_subset=feature_subset)
    dat_loader = data_utils.DataLoader(dset_test, batch_size=1, shuffle=False, collate_fn=dat.collate_fn_padd_local)

    df=[]
    for batch_idx, batch in enumerate(dat_loader):
        x, mask, gene_indicators, sv_indicators, y = batch
        prediction, truth, element_predictions = model.get_element_score(batch)

        file = np.load(test_df['PATH'][batch_idx])
        element_names = file['element_names'].tolist()
        if len(element_predictions)>len(element_names):
            element_predictions = element_predictions[1:-1]
        element_dict = {'ID': test_df['ID'][batch_idx], 'sv_prediction': prediction, 'sv_label': truth,
                        'element_names': element_names,'element_predictions':element_predictions}#'overall_score': overall_score}
        df_ = pd.DataFrame(element_dict)
        df.append(df_)
    df = pd.concat(df)
    df.to_csv(dest_file)




#https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning

##########################################relate with predictions##########################################
def calibrate_confidence(confidence, cutoff):
    logit_bias = np.log(cutoff / (1 - cutoff))
    logit_cali = [np.log(c/(1-c))-logit_bias for c in confidence]
    confidence_cali = [1/(1+np.exp(-l))for l in logit_cali]
    return confidence_cali
def get_global_genes(chr,start,end,elements_path,tad_path=None):
    """
    tad_path: put None if query all genes in 1000k flanking region; put tad_path or tad bed file for genes within tad
    elements_path: put element_path or bw element file
    output: 1. gene names 2. intronic gene names
    """
    if isinstance(elements_path,str):
        elements = pyBigWig.open(elements_path, "r")
    else:
        elements = elements_path
    sv_tri = [chr,start,end]
    sv_bed = u.tri_to_bed(sv_tri)
    if tad_path is not None:
        #use tad
        if isinstance(tad_path,str):
            _, _, tad = u.read_bed(tad_path, N=None, parse=False)
        else:
            tad = tad_path
        tad_r = tad.loc[tad.iloc[:, 3] == "0", :2]
        tad_b = tad.loc[tad.iloc[:, 3] == "1", :2]
        tad_region_tri = rf.superset_tad(sv_tri, tad_r, tad_b)
    else:
        #flanking
        sv_flank = sv_bed.flank(genome='hg38', b=1000000).to_dataframe()
        tad_region_tri = [sv_tri[0], int(min(sv_flank.start.min(),sv_tri[1])), int(max(sv_flank.end.max(),sv_tri[2]))]
    _, elements_coords_bed, _, gene_names = rf.tad_elements(elements, tad_region_tri)
    #intronic
    _, _, _, intronic_gene_names = rf.tad_elements(elements, sv_tri)
    return gene_names, intronic_gene_names



##breakpoint
def bke_to_sv(CHR, BKE, ID, elements_path, annotation_path,svtype='insertion'):
    start = BKE
    end = int(start+1)
    anno = None
    if ID is None:
        ID='ID'
    if isinstance(elements_path,str):
        elements = pyBigWig.open(elements_path, "r")
    else:
        elements = elements_path
    if svtype=='insertion':
        sv_list = [[CHR, int(start - 50), int(start + 50), 'deletion', ID]]
    elif svtype=='inversion':
        anno = u.annot_sv_single(CHR, start, end, annotation_path)
        if anno == 'noncoding':
            _, _, gene_indicator, _ = rf.tad_elements(elements, [CHR, start, end])
            gene_indicator = np.sum(gene_indicator)
            if gene_indicator > 0:
                anno = 'intron'
        if anno == 'coding' or anno=='intron':
            elements_coords_df, _, _, _ = rf.tad_elements(elements, [CHR, start, end])
            elements_coords_df = elements_coords_df[elements_coords_df.iloc[:,3]!='noncoding']
            starts = elements_coords_df.iloc[:,1].tolist()
            ends = elements_coords_df.iloc[:, 2].tolist()
            sv_list = []
            for (s, e) in zip(starts, ends):
                sv_list.append([CHR, s, end, 'deletion',ID,'left'])
                sv_list.append([CHR, start, e, 'deletion',ID,'right'])
        else:
            sv_list = [[CHR, int(start - 50), int(start + 50), 'deletion', ID,'noncoding']]
    else:#translocation
        anno = u.annot_sv_single(CHR, start, end, annotation_path)
        if anno == 'noncoding':
            _, _, gene_indicator, _ = rf.tad_elements(elements, [CHR, start, end])
            gene_indicator = np.sum(gene_indicator)
            if gene_indicator > 0:
                anno = 'intron'
        elements_coords_df, _, _, _ = rf.tad_elements(elements, [CHR, start, end])
        starts = elements_coords_df.iloc[:, 1].tolist()
        ends = elements_coords_df.iloc[:, 2].tolist()
        sv_list = []
        for (s, e) in zip(starts, ends):
            sv_list.append([CHR, s, end, 'deletion', ID,'left'])
            sv_list.append([CHR, start, e, 'deletion', ID,'right'])
    sv_df = pd.DataFrame(sv_list)
    if sv_df.shape[1]==5:
        sv_df.columns = ['CHR','START','END','SVTYPE','ID']
    else:
        sv_df.columns = ['CHR', 'START', 'END', 'SVTYPE', 'ID','TRUNCATION']
    return  sv_df, anno

def sv_transformation(CHR, START, END, SVTYPE, ID,elements_path, annotation_path,
                      CHR2=None, START2=None, strand1='+', strand2='-',full_mode=False):
    if ID is None:
        ID='ID'
    sv_df = None
    # insertion---some of the SVs might be delin
    if SVTYPE=='insertion':
        # delin---treat as deletion
        if END-START>1:
            sv_df = pd.DataFrame({'CHR': [CHR], 'START': [START], 'END': [END], 'SVTYPE': ['deletion'], 'ID': [ID]})
        else:
            sv_df, _ = bke_to_sv(CHR,START,ID,elements_path, annotation_path,'insertion')
    ##full mode: set to false when considering coding as priority
    if SVTYPE=='inversion':
        if u.annot_sv_single(CHR, START, END, annotation_path)=='noncoding':
            df1, anno1 = bke_to_sv(CHR, START, ID, elements_path, annotation_path,'insertion')
            df2, anno2 = bke_to_sv(CHR, END, ID, elements_path, annotation_path,'insertion')
        else:
            df1, anno1 = bke_to_sv(CHR, START, ID, elements_path, annotation_path,'inversion')
            if df1.shape[1]==6:
                df1 = df1[df1.TRUNCATION=='right'].iloc[:,:5]
            df2, anno2 = bke_to_sv(CHR, END, ID, elements_path, annotation_path,'inversion')
            if df1.shape[1]==6:
                df1 = df2[df2.TRUNCATION=='right'].iloc[:,:5]
        #at least one end impact coding (we consider the impact of disrupting coding seq)
        if full_mode:
            sv_df = pd.concat([df1, df2]).reset_index(drop=True)  # both coding or both noncoding
        else:
            if anno1 in ['coding','intron'] and anno2=='noncoding':
                sv_df = df1
            elif anno2 in ['coding','intron']and anno1=='noncoding':
                sv_df=df2
            else:
                sv_df = pd.concat([df1, df2]).reset_index(drop=True) #both coding or both noncoding
    if SVTYPE=='translocation':
        df1, anno1 = bke_to_sv(CHR, START, ID, elements_path, annotation_path,'translocation')
        if strand1 =='+':
            truncation1 = 'right'
        else:
            truncation1 = 'left'
        df1 = df1[df1.TRUNCATION == truncation1]
        if CHR2 is not None:
            df2, anno2 = bke_to_sv(CHR2, START2, ID, elements_path, annotation_path, 'translocation')
            if anno1 in ['coding','intron'] and anno2 in ['coding','intron']:
                df2['SVTYPE'] = ['duplication']*len(df2['SVTYPE'])
                if strand2 == '+': #keep
                    truncation2 = 'right'
                else:
                    truncation2 = 'left'
            else:
                if strand2 == '+': #truncate
                    truncation2 = 'left'
                else:
                    truncation2 = 'right'
            df2 = df2[df2.TRUNCATION == truncation2]
            sv_df = pd.concat([df1, df2]).reset_index(drop=True)
        else:
            sv_df = df1
    if SVTYPE=='deletion' or SVTYPE=='duplication':
        sv_df = pd.DataFrame({'CHR': [CHR], 'START': [START], 'END': [END], 'SVTYPE': ['deletion'], 'ID': [ID]})
    return sv_df


#coding model features
def read_local_features_singlesv(chr, start, end, svtype, elements_path, feature_files, scaler_file, return_coords=False,
                                 feature_subset=None):
    """
    read features of a given coding SV
    elements_path: put element_path or bw element file
    output: 1. batch 2. element names
    """
    if isinstance(elements_path,str):
        elements = pyBigWig.open(elements_path, "r")
    else:
        elements = elements_path
    sv_tri = [chr, start, end]
    if svtype == 'DEL':
        svtype = 'deletion'
    if svtype == 'DUP':
        svtype = 'duplication'
    assert svtype == 'deletion' or svtype == 'duplication', f'{svtype} is not supported'
    # get elements
    elements_coords_df, elements_coords_bed, gene_indicator, gene_names = rf.tad_elements(elements, sv_tri)
    # intersect with elements
    sv_region = ' '.join([str(i) for i in sv_tri])
    sv_region_bed = BedTool(sv_region, from_string=True)
    elements_coords_df = elements_coords_bed.intersect(sv_region_bed).to_dataframe()
    _, features, _, _ = rf.read_features(elements_coords_df, feature_files=feature_files)
    features = rf.scale_features(features, scaler_file)
    if svtype == 'deletion':
        features = np.concatenate((features, np.zeros((features.shape[0], features.shape[1], 1))), axis=-1)
    else:
        features = np.concatenate((features, np.ones((features.shape[0], features.shape[1], 1))), axis=-1)
    features = np.squeeze(features,1)
    name_list = [n.replace('_sv','') for n in elements_coords_df['name'].tolist()]
    gene_indicators = np.expand_dims(np.array([0 if e in ['noncoding', 'sv'] else 1 for e in name_list]), 0)
    element_names = elements_coords_df['name'].tolist()
    features = np.expand_dims(features, 0)
    if feature_subset is not None:
        features = features[..., feature_subset]
    mask = np.ones(features.shape)
    sv_indicators = np.ones(gene_indicators.shape)
    features = torch.tensor(features).float()
    mask = torch.tensor(mask).float()
    gene_indicators = torch.tensor(gene_indicators).float()
    sv_indicators = torch.tensor(sv_indicators).float()
    y_plackholder = torch.tensor(0).float()
    batch = (features, mask, gene_indicators, sv_indicators, y_plackholder)
    if return_coords:
        return batch, element_names, elements_coords_df
    else:
        return batch, element_names
#noncoding model features
def read_global_features_singlesv(chr, start, end, svtype, elements_path, feature_files, scaler_file, tad_path=None, return_coords=False,
                                  truncation = None, feature_subset=None):
    """
    read features of a given noncoding SV
    elements_path: put element_path or bw element file
    output: 1. batch 2. element names
    only support deletion and duplication for now
    """
    if isinstance(elements_path,str):
        elements = pyBigWig.open(elements_path, "r")
    else:
        elements = elements_path
    sv_tri = [chr, start, end]
    if svtype == 'DEL':
        svtype = 'deletion'
    if svtype == 'DUP':
        svtype = 'duplication'
    assert svtype == 'deletion' or svtype == 'duplication' , f'{svtype} is not supported'
    sv_bed = u.tri_to_bed(sv_tri)
    # get global region: tad_region_tri
    if tad_path is not None:
        # use tad
        if isinstance(tad_path, str):
            _, _, tad = u.read_bed(tad_path, N=None, parse=False)
        else:
            tad = tad_path
        tad_r = tad.loc[tad.iloc[:, 3] == "0", :2]
        tad_b = tad.loc[tad.iloc[:, 3] == "1", :2]
        tad_region_tri = rf.superset_tad(sv_tri, tad_r, tad_b)
    else:
        # flanking
        sv_flank = sv_bed.flank(genome='hg38', b=1000000).to_dataframe()
        tad_region_tri = [sv_tri[0], int(min(sv_flank.start.min(),sv_tri[1])), int(max(sv_flank.end.max(),sv_tri[2]))]
    elements_coords_df, elements_coords_bed, _, gene_names = rf.tad_elements(elements, tad_region_tri)
    elements_start = elements_coords_bed.to_dataframe().start.min()
    elements_end = elements_coords_bed.to_dataframe().end.max()
    #########
    if sv_tri[1] - tad_region_tri[1] > 1 and truncation!='left':
        left_bed = u.tri_to_bed([elements_coords_bed[0].chrom, elements_start, sv_bed[0].start])
        df_left = elements_coords_bed.intersect(left_bed).to_dataframe()
        df_left = df_left[df_left.name!='noncoding'].reset_index(drop=True)
        if df_left.shape[0]==0:
            df_left = None
    else:
        df_left = None
    if tad_region_tri[2] - sv_tri[2] > 1 and truncation!='right':
        right_bed = u.tri_to_bed([elements_coords_bed[0].chrom, sv_bed[0].end, elements_end])
        df_right = elements_coords_bed.intersect(right_bed).to_dataframe()
        df_right = df_right[df_right.name != 'noncoding'].reset_index(drop=True)
        if df_right.shape[0] == 0:
            df_right = None
    else:
        df_right = None
    df_sv = elements_coords_bed.intersect(sv_bed).to_dataframe()
    ##########
    # get feature array
    # ----left
    if df_left is not None:
        _, feature_array_left, _, element_name_left = rf.read_features(df_left, feature_files=feature_files)
        if scaler_file is not None:
            feature_array_left = rf.scale_features(feature_array_left, scaler_file=scaler_file)
        feature_array_left = np.concatenate((feature_array_left, 0.5 + np.zeros((feature_array_left.shape[0], feature_array_left.shape[1], 1))),axis=-1)
    else:
        feature_array_left = None
        element_name_left = None
    # ----sv
    _, feature_array_sv, _, element_name_sv = rf.read_features(df_sv, feature_files=feature_files)
    element_name_sv_suffix = [e + '_sv' for e in element_name_sv]
    if scaler_file is not None:
        feature_array_sv = rf.scale_features(feature_array_sv, scaler_file=scaler_file)
    if svtype == 'deletion':
        feature_array_sv = np.concatenate((feature_array_sv, np.zeros((feature_array_sv.shape[0], feature_array_sv.shape[1], 1))), axis=-1)
    else:# svtype == 'duplication':
        feature_array_sv = np.concatenate((feature_array_sv, np.ones((feature_array_sv.shape[0], feature_array_sv.shape[1], 1))), axis=-1)
    # ----right
    if df_right is not None:
        _, feature_array_right, _, element_name_right = rf.read_features(df_right, feature_files=feature_files)
        if scaler_file is not None:
            feature_array_right = rf.scale_features(feature_array_right, scaler_file=scaler_file)
        feature_array_right = np.concatenate((feature_array_right, 0.5 + np.zeros((feature_array_right.shape[0], feature_array_right.shape[1], 1))),axis=-1)
    else:
        feature_array_right = None
        element_name_right = None
    #merge
    array_list = [f for f in [feature_array_left, feature_array_sv, feature_array_right] if f is not None]
    features = np.concatenate(array_list, axis=0)
    features = np.squeeze(features, 1)
    element_name_list = [element_name_left, element_name_sv_suffix, element_name_right]
    element_name_list = [ele for ele in element_name_list if ele is not None]
    element_names = u.unpack_list(element_name_list)
    element_name_list_ = [element_name_left, element_name_sv, element_name_right]
    element_name_list_ = [ele for ele in element_name_list_ if ele is not None]
    element_name_ = u.unpack_list(element_name_list_)
    gene_indicators = [1 if e in gene_names else 0 for e in element_name_]  # intron is accounted as genes
    gene_indicators = np.expand_dims(np.array(gene_indicators),0)
    features = np.expand_dims(features, 0)
    if feature_subset is not None:
        features = features[..., feature_subset]
    mask = np.ones(features.shape)
    sv_indicators = np.expand_dims(np.array([1 if 'sv' in e else 0 for e in element_names]),0)
    features = torch.tensor(features).float()
    mask = torch.tensor(mask).float()
    gene_indicators = torch.tensor(gene_indicators).float()
    sv_indicators = torch.tensor(sv_indicators).float()
    y_plackholder = torch.tensor(0).float()
    batch = (features, mask, gene_indicators, sv_indicators, y_plackholder)
    if return_coords:
        return batch, element_names, (df_left,df_sv,df_right)
    else:
        return batch, element_names

def prepare_model(ckpt_path):
    model = mm.LocalSVModule.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def predict_codingsv(chr, start, end, svtype, elements_path, feature_files, scaler_file, model, calibration_cutoff=0.5, feature_subset=None):
    # prepare features and element names
    batch, element_names = read_local_features_singlesv(chr, start, end, svtype, elements_path, feature_files, scaler_file, feature_subset=feature_subset)
    # get prediction
    sv_pred, _, element_pred = model.get_element_score(batch)
    if len(element_pred)>len(element_names):
        element_pred = element_pred[1:-1]
    element_names = ['SV'] + element_names
    type_indicator = ['Exonic']*len(element_names)
    type_indicator[0] = 'Coding SV'
    sv_pred = [sv_pred] + element_pred
    sv_pred_cali = calibrate_confidence(sv_pred, cutoff=calibration_cutoff)
    #get results
    df = pd.DataFrame({'Elements': element_names, 'Pathogenicity': sv_pred_cali, 'Type':type_indicator})
    return df

def predict_noncodingsv(chr,start, end, svtype,elements_path,feature_files,scaler_file,model,tad_path=None, calibration_cutoff=0.5, truncation=None, feature_subset=None):
    #prepare features and element names
    batch,element_names = read_global_features_singlesv(chr,start, end, svtype,elements_path,feature_files,scaler_file,tad_path, False,truncation, feature_subset=feature_subset)
    gene_names,intronic_gene_names = get_global_genes(chr,start,end,elements_path,tad_path)
    sv_pred, _, element_pred = model.get_element_score(batch)
    if len(element_pred)>len(element_names):
        element_pred = element_pred[1:-1]
    element_names = [e.replace('_sv','') if '_sv' in e else e for e in element_names]
    df = pd.DataFrame({'element_names': element_names, 'element_pred': element_pred})
    df=df[df.element_names!='noncoding'].reset_index(drop=True)
    df = df[df.element_names != 'sv'].reset_index(drop=True)
    df = df.groupby('element_names').agg('max')
    element_names = ['SV'] + df.index.tolist()
    type_indicator = ['Intronic' if g in intronic_gene_names else 'Regulatory' for g in element_names]
    type_indicator[0]='Non-coding SV'
    sv_pred =[sv_pred]+df.element_pred.tolist()
    sv_pred_cali = calibrate_confidence(sv_pred, cutoff=calibration_cutoff)
    #get results
    df=pd.DataFrame({'Elements': element_names, 'Pathogenicity': sv_pred_cali, 'Type':type_indicator})
    return df



###############################predict new sv###################################

def phenosv(CHR, START, END, svtype,sv_df=None, annotation_path=None, model=None, elements_path=None, feature_files=None, scaler_file=None,
            tad_path=None,cutoff_coding=0.4934, cutoff_noncoding=0.7736, HPO=None, pheno_adjust=1,KBpath='/home/xu3/Phen2Gene/lib', full_mode = False,
            CHR2=None, START2=None,strand1='+',strand2='+', feature_subset=None):
    if svtype in ['insertion','inversion']:
        sv_df = pd.DataFrame({'CHR':[CHR],'START':[START],'END':[END],'SVTYPE':[svtype],'ID':[CHR+':'+str(START)+'-'+str(END)+'_'+str(svtype)]})
    if svtype=='translocation':
        sv_df = pd.DataFrame({'CHR1':[CHR],'START1':[START],'END1':[END],'STRAND1':[strand1],
                              'CHR2':[CHR2],'START2':[START2],'END2':[START2],'STRAND2':[strand2],
                              'SVTYPE':['translocation'],'ID':[CHR+':'+str(START)+'-'+CHR2+':'+str(START2)]})
    if sv_df is not None:
        #translocation to standard sv_df
        if 'CHR2' in sv_df.columns.tolist():
            sv_list = []
            for i in range(sv_df.shape[0]):
                CHR1, START1, END1, STRAND1 = sv_df['CHR1'][i], sv_df['START1'][i], sv_df['END1'][i], sv_df['STRAND1'][i]
                CHR2, START2, END2, STRAND2 = sv_df['CHR2'][i], sv_df['START2'][i], sv_df['END2'][i], sv_df['STRAND2'][i]
                svtype, ID = 'translocation', sv_df['ID'][i]
                sv = sv_transformation(CHR1, START1, END1, svtype, ID, elements_path, annotation_path,
                                       CHR2, START2, STRAND1, STRAND2,full_mode=False)
                sv_list.append(sv)
            sv_df = pd.concat(sv_list).reset_index(drop=True)
        sv_list = []
        for i in range(sv_df.shape[0]):
            CHR, START, END, svtype,ID = sv_df['CHR'][i], sv_df['START'][i], sv_df['END'][i], sv_df['SVTYPE'][i],sv_df['ID'][i]
            sv = sv_transformation(CHR, START, END, svtype,ID,elements_path, annotation_path, None, None, full_mode = full_mode)
            if sv.shape[1]==5:
                sv['TRUNCATION']=None
            sv_list.append(sv)
        sv_df_append = pd.concat(sv_list).reset_index(drop=True)
        if 'HPO' in sv_df.columns.tolist():
            sv_df_append = sv_df_append.merge(sv_df[['ID','HPO']],how='left').reset_index(drop=True)
        pred = multi_sv(sv_df_append, annotation_path, model, elements_path, feature_files, scaler_file, tad_path=tad_path,
                 cutoff_coding=cutoff_coding, cutoff_noncoding=cutoff_noncoding, HPO=HPO, pheno_adjust=pheno_adjust, full_mode=full_mode,
                 KBpath=KBpath, feature_subset=feature_subset)
        pred = pred.reset_index(drop=True)
        idx = pred.groupby(['ID', 'Elements'], sort=False)['Pathogenicity'].idxmax()
        pred = pred.loc[idx].reset_index(drop=True)
    else:
        pred = single_sv(CHR, START, END, svtype, annotation_path, model, elements_path, feature_files, scaler_file,
                  tad_path=tad_path,cutoff_coding=cutoff_coding, cutoff_noncoding=cutoff_noncoding, HPO=HPO,
                  pheno_adjust=pheno_adjust, KBpath=KBpath,
                  full_mode=full_mode, truncation=None, feature_subset=feature_subset)
    return pred

def single_sv(CHR, START, END, svtype, annotation_path, model, elements_path, feature_files, scaler_file,tad_path=None,
              cutoff_coding=0.5, cutoff_noncoding=0.5, HPO=None, pheno_adjust=1,KBpath='/home/xu3/Phen2Gene/lib',
              full_mode = False,truncation=None, feature_subset=None):
    """
    annotation_path: can be a path or bed file
    """
    if u.annot_sv_single(CHR, START, END, annotation_path) == 'coding':
        pred = predict_codingsv(CHR, START, END, svtype, elements_path, feature_files,scaler_file, model, cutoff_coding, feature_subset=feature_subset)
        if full_mode:
            pred2 = predict_noncodingsv(CHR, START, END, svtype, elements_path, feature_files,scaler_file, model, tad_path, cutoff_noncoding,feature_subset=feature_subset)
            type=[]
            for t in pred2.Type.tolist():
                if t=='Intronic':
                    type.append('Intronic-like')
                elif t=='Non-coding SV':
                    type.append('Coding SV indirect')
                else:
                    type.append(t)
            pred2.Type = type
            pred = pd.concat((pred,pred2))
    else:
        pred = predict_noncodingsv(CHR, START, END, svtype, elements_path, feature_files,scaler_file, model, tad_path, cutoff_noncoding,truncation=truncation,feature_subset=feature_subset)
    if HPO is not None:
        genescores = pg.phen2gene(HPO,KBpath, scale_score=True)
        if genescores is not None:
            imp_min = genescores.Score.min()
            pred = pred.merge(genescores, left_on='Elements', right_on='Gene', how='left')
            pred = pred[['Elements', 'Pathogenicity', 'Type', 'Score']]
            pred = pred.fillna(imp_min)
            imp_max = pred.Score.max()
        else:
            imp_max = 0
            pred['Score'] = [0] * pred.shape[0]
            pred = pred[['Elements', 'Pathogenicity', 'Type', 'Score']]
        scores = [imp_max if pred['Elements'][i] == 'SV' else pred['Score'][i] for i in range(pred.shape[0])]
        pred['Score'] = scores
        cn = pred.columns.tolist()
        cn[-1] = 'Phen2Gene'
        pred.columns = cn
        pred['PhenoSV'] = pred.Pathogenicity * pow(pred.Phen2Gene, pheno_adjust)
    pred = pred[pred['Type'] != 'Intronic-like'].reset_index(drop=True)
    pred = pred[pred['Elements'] != 'noncoding'].reset_index(drop=True)
    return pred

def multi_sv(sv_df, annotation_path, model,elements_path, feature_files, scaler_file,tad_path=None,
              cutoff_coding=0.5, cutoff_noncoding=0.5, HPO=None, pheno_adjust=1, full_mode=False,
             KBpath='/home/xu3/Phen2Gene/lib',feature_subset=None):
    #HPO can be None, a string representing the same HPO for all SV, or a list representing different HPO for different SV
    pred_list=[]
    if HPO is not None:
        if isinstance(HPO, str):
            HPO = [HPO]*sv_df.shape[0]
    else:
        HPO = [None]*sv_df.shape[0]
    for i in range(sv_df.shape[0]):
        CHR, START, END, svtype = sv_df['CHR'][i], sv_df['START'][i], sv_df['END'][i], sv_df['SVTYPE'][i]
        if 'TRUNCATION' in sv_df.columns.tolist():
            truncation = sv_df['TRUNCATION'][i]
        else:
            truncation = None
        pred = single_sv(CHR, START, END, svtype, annotation_path, model, elements_path, feature_files, scaler_file,tad_path=tad_path,
              cutoff_coding=cutoff_coding, cutoff_noncoding=cutoff_noncoding, HPO=HPO[i], pheno_adjust=pheno_adjust, KBpath=KBpath,
                         full_mode=full_mode,truncation=truncation,feature_subset=feature_subset)
        pred['ID'] = sv_df['ID'][i]
        pred_list.append(pred)
    pred = pd.concat(pred_list)
    return pred







##########################################model interpretability ##########################################

def input_gradient(CHR, START, END, SVTYPE, elements_path, feature_files, scaler_file, ckpt_path, annotation_path,
                   target_file_name = None, target_folder=None, tad_path=None, force_noncoding=False):
    model = prepare_model(ckpt_path)
    if u.annot_sv_single(CHR, START, END, annotation_path) == 'coding' and force_noncoding==False:
        batch, element_names, coords = read_local_features_singlesv(CHR, START, END, SVTYPE, elements_path, feature_files,scaler_file, return_coords=True)
        cutoff = 0.4934
    else:
        batch, element_names , coords = read_global_features_singlesv(CHR, START, END, SVTYPE, elements_path, feature_files, scaler_file,tad_path, return_coords=True)
        cutoff = 0.7736

    sv_pred, _, element_pred = model.get_element_score(batch)
    if len(element_pred) > len(element_names):
        element_pred = element_pred[1:-1]
    element_pred = calibrate_confidence(element_pred, cutoff)
    prediction_df = pd.DataFrame({"element_names":element_names,"element_pred":element_pred})
    #sv prediction
    x, mask, gene_indicator, sv_indicator, _ = batch
    x.requires_grad_()
    prediction = model(x, mask, gene_indicator, sv_indicator)
    prediction[1].backward()
    gradient_map = x.grad
    input_gradient_map = gradient_map*x
    gradient_map = gradient_map.detach().numpy()
    input_gradient_map = input_gradient_map.detach().numpy()
    if target_folder is not None:
        if target_file_name is not None:
            target_file_path = os.path.join(target_folder, target_file_name+'_input_gradient.npz')
        else:
            target_file_path = os.path.join(target_folder, 'input_gradient.npz')
        np.savez(target_file_path, gradient_map=gradient_map, input_gradient_map=input_gradient_map,
                 element_names=element_names, prediction_df=prediction_df, coords=coords)
    else:
        return gradient_map, input_gradient_map, element_names, prediction_df, coords

def attention_map(CHR, START, END, SVTYPE, elements_path, feature_files, scaler_file, ckpt_path, annotation_path,
                   target_file_name = None, target_folder=None, tad_path=None, force_noncoding=False):
    model = prepare_model(ckpt_path)
    if u.annot_sv_single(CHR, START, END, annotation_path) == 'coding' and force_noncoding==False:
        batch, element_names = read_local_features_singlesv(CHR, START, END, SVTYPE, elements_path, feature_files,scaler_file)
    else:
        batch, element_names = read_global_features_singlesv(CHR, START, END, SVTYPE, elements_path, feature_files,
                                                             scaler_file, tad_path)
    x, _, gene_indicator, sv_indicator, _ = batch
    attention_maps = model.get_attention_maps(x, gene_indicator, sv_indicator)
    attention_maps = [att.detach().numpy() for att in attention_maps]
    attention_maps = np.concatenate(attention_maps,0)
    if target_folder is not None:
        if target_file_name is not None:
            target_file_path = os.path.join(target_folder, target_file_name+'_attention_map.npz')
        else:
            target_file_path = os.path.join(target_folder, 'attention_map.npz')
        np.savez(target_file_path, attention_maps=attention_maps, element_names=element_names)
    else:
        return attention_maps, element_names

