import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)
import numpy as np
from utilities import utility
import pandas as pd
from pybedtools import BedTool
import pyBigWig
import model.operation_function as of

def read_features_bw(sv_tri, feature_path, stat = 'max',nbins=1, onlycoverage = 1, binlen = None):
    if isinstance(stat, list)==False:
        stat=[stat]
    if len(stat)==1:
        stat=stat*len(feature_path)
    if isinstance(onlycoverage, list)==False:
        onlycoverage=[onlycoverage]
    if len(onlycoverage)==1:
        onlycoverage=onlycoverage*len(feature_path)
    feature_list=[]
    for i, path in enumerate(feature_path):
        output = utility.query_bw(path,chrom = sv_tri[0], start = int(sv_tri[1]), end =int(sv_tri[2]),
                    nbins=nbins, stat=stat[i], onlycoverage= onlycoverage[i], binlen= binlen)
        feature_list.append(output)
    feature_list = np.array(feature_list).transpose()
    return feature_list #bin *feature

def read_features_bb(sv_tri, feature_path, categories, stat = 'max',nbins=1, binlen=None):
    feature_list=[]
    for i in range(len(feature_path)):
        path = feature_path[i]
        category = categories[i]
        output = utility.query_bb(path,chrom = sv_tri[0], start = int(sv_tri[1]), end =int(sv_tri[2]),
                    nbins=nbins, stat=stat, categories=category, binlen=binlen)
        feature_list.append(output)
    feature_list = np.concatenate(feature_list,1)
    return feature_list #bin *feature


def read_features(sv_df, feature_files='../../data/Features/featuremaster.csv'):
    features_df = pd.read_csv(feature_files)
    #bw
    feature_name_bw = features_df.loc[features_df['type']=='bigwig','feature'].tolist()
    feature_path_bw = features_df.loc[features_df['type']=='bigwig','path'].tolist()
    feature_mean = features_df.loc[features_df['type'] == 'bigwig', 'mean'].tolist()
    feature_mean = [int(f) for f in feature_mean]
    feature_max = features_df.loc[features_df['type'] == 'bigwig', 'max'].tolist()
    feature_max = [int(f) for f in feature_max]
    feature_sum = features_df.loc[features_df['type'] == 'bigwig', 'sum'].tolist()
    feature_sum = [int(f) for f in feature_sum]
    #update bw names
    feature_name_bw = [fn + '_mean' for i, fn in enumerate(feature_name_bw) if feature_mean[i] == 1] + \
                      [fn + '_max' for i, fn in enumerate(feature_name_bw) if feature_max[i] == 1]+ \
                      [fn + '_min' for i, fn in enumerate(feature_name_bw) if feature_max[i] == 2] + \
                      [fn + '_sum' for i, fn in enumerate(feature_name_bw) if feature_sum[i] == 1]
    feature_stat = [fn.split('_')[-1] for fn in feature_name_bw]
    #update bw paths
    feature_path_bw = [feature_path_bw[i] for i, f in enumerate(feature_mean) if f == 1] + \
                      [feature_path_bw[i] for i, f in enumerate(feature_max) if f == 1] + \
                      [feature_path_bw[i] for i, f in enumerate(feature_max) if f == 2] + \
                      [feature_path_bw[i] for i, f in enumerate(feature_sum) if f == 1]
    feature_onlycoverage_ = features_df.loc[features_df['type'] == 'bigwig', 'onlycoverage'].tolist()
    feature_onlycoverage_ = [feature_onlycoverage_[i] for i, f in enumerate(feature_mean) if f == 1]
    feature_onlycoverage = feature_onlycoverage_ + \
                           [0] * (len(feature_path_bw) - len(feature_onlycoverage_))
    #bb
    feature_name_bb = features_df.loc[features_df['type'] == 'bigbed', 'feature'].tolist()
    feature_path_bb = features_df.loc[features_df['type'] == 'bigbed', 'path'].tolist()
    categories_bb = features_df.loc[features_df['type'] == 'bigbed', 'categories'].tolist()
    categories_bb = [int(c) for c in categories_bb]
    #update bb names
    name_list = []
    for i in range(len(categories_bb)):
        name_list.append([feature_name_bb[i] + '_' + str(cat) for cat in list(range(categories_bb[i]))])
    feature_name_bb = utility.unpack_list(name_list)
    #overall files
    feature_names = feature_name_bw + feature_name_bb

    sv_list = sv_df.iloc[:,:3].values.tolist()
    sv_names = sv_df.iloc[:,3].tolist()
    sv_name_list = []
    sv_feature_list = []
    error_list = []

    for i in range(len(sv_list)):
        sv_ = sv_list[i]
        try:
            if len(feature_path_bb)>0 and len(feature_path_bw)>0:
                sv_feature_bw = read_features_bw(sv_, feature_path_bw, stat = feature_stat, nbins=1, onlycoverage=feature_onlycoverage, binlen=None) #bin * feature
                sv_feature_bb = read_features_bb(sv_, feature_path_bb, categories = categories_bb, stat='sum', nbins=1,binlen=None)  # bin * feature
                sv_features = np.concatenate([sv_feature_bw,sv_feature_bb],1)
            elif len(feature_path_bw)==0 and len(feature_path_bb)>0:
                sv_features = read_features_bb(sv_, feature_path_bb, categories = categories_bb, stat='sum', nbins=1,binlen=None)
            else:
                sv_features = read_features_bw(sv_, feature_path_bw, stat = feature_stat, nbins=1, onlycoverage=feature_onlycoverage,binlen=None)
            sv_feature_list.append(sv_features)
            sv_name_list.append(sv_names[i])
        except (RuntimeError, TypeError, NameError):
            print (i)
            error_list.append(i)
    return feature_names, np.array(sv_feature_list), error_list, sv_name_list  # sv*bin*feature



def superset_tad(query_region, tad_r, tad_b):
    '''
    get genome regions where adjacent TADs can cover given SV
    :param: query_region: example: query_region = 'chr1 80385 91719' or as a tri ['chr1', 80385, 91719]
    :param: tad_r: data frame of tad region
    :param: tad_b: data frame of tad boundaries
    :return: a list of genome region; example: ['chr1', 60000, 684620]
    '''
    if isinstance(query_region, str)==False:
        query_region = ' '.join([str(i) for i in query_region])
    query_region_bed = BedTool(query_region, from_string=True)
    tad_r_bed = BedTool.from_dataframe(tad_r)
    tad_b_bed = BedTool.from_dataframe(tad_b)
    # boudary intersect
    inter_b = tad_b_bed.intersect(query_region_bed, wa=True, wb=True).to_dataframe()
    if inter_b.shape[0] == 0:
        inter_r = tad_r_bed.intersect(query_region_bed, wa=True).to_dataframe()
        region = inter_r.values.tolist()[0]
    else:
        boundary_selected = ' '.join([inter_b.chrom[0], str(min(inter_b.start)), str(max(inter_b.end))])
        boundary_selected_bed = BedTool(boundary_selected, from_string=True)
        fu = boundary_selected_bed.closest(tad_r_bed, D='ref', fu=True).to_dataframe()
        fd = boundary_selected_bed.closest(tad_r_bed, D='ref', fd=True).to_dataframe()
        region = [fu.values[0][0], fu.values[0][4], fd.values[0][5]]
    return region


def tad_elements(element_bb, tad_region_tri):
    #elements_path = '/Volumes/G-DRIVE USB-C/GenomeFeatures/FeatureSet/elements_w_nonelement.bb'
    #elements_bb = pyBigWig.open(elements_path, "r")
    '''
    get elements within tad genome region
    :param: element_bb: object of element generated by pyBigWig: example: elements_bb = pyBigWig.open(elements_path, "r")
    :param: tad_region_tri: tad region, example: ['chr1', 60000, 684620]
    :return: df of elements, bed format of elements
    '''

    elements_coords = element_bb.entries(tad_region_tri[0], int(tad_region_tri[1]), int(tad_region_tri[2]))
    gene_names = []
    gene_indicator = []
    elements_coords_ = []

    for ele in elements_coords:
        elements_coords_.append([tad_region_tri[0]] + list(ele[:2]) + [ele[2].split('\t')[0]])
        gene_indicator_pair = ele[2].split('\t')
        gene_indicator.append(int(gene_indicator_pair[1]))
        if gene_indicator_pair[1]=='1':
            gene_names.append(gene_indicator_pair[0])

    elements_coords_df = pd.DataFrame(elements_coords_)
    elements_coords_bed = BedTool.from_dataframe(elements_coords_df)
    return elements_coords_df, elements_coords_bed, gene_indicator, gene_names








def scale_features(feature_array, scaler_file = '/Users/karenxu/Documents/Project_PhenoSV/data/Features/features0705.csv'):
    if '.csv' in scaler_file:
        scaler = pd.read_csv(scaler_file)
        feature_array = (feature_array - np.array(scaler['bias'])) / np.array(scaler['scale'])
    #npy file for percentile norm
    else:
        reference = np.load(scaler_file).squeeze()
        if len(feature_array.shape)>2:
            feature_array = feature_array.squeeze()
        if len(feature_array.shape)==1:
            feature_array = feature_array[None,:]
        feature_array = utility.percentile_rank(reference, feature_array)
        feature_array = np.expand_dims(feature_array,axis=1)
    return feature_array




def save_sv_features_local(sv_df, target_folder,feature_files,elements_path, scaler_file,
                           aggregate_sv=False, force_region=False):
    elements = pyBigWig.open(elements_path, "r")
    for i in range(sv_df.shape[0]):
        sv_tri = sv_df.iloc[i, :3].tolist()
        sv_type = sv_df.iloc[i, 4]
        sv_name = sv_df.iloc[i, 3]
        target_file_path = os.path.join(target_folder, sv_name + '.npz')
        assert sv_type=='deletion' or sv_type=='duplication', f'{sv_type} is not supported'
        #get elements
        _, elements_coords_bed,gene_indicator, gene_names = tad_elements(elements, sv_tri)
        #intersect with elements
        sv_region = ' '.join([str(i) for i in sv_tri])
        sv_region_bed = BedTool(sv_region, from_string=True)
        elements_coords_df = elements_coords_bed.intersect(sv_region_bed).to_dataframe()
        if aggregate_sv: #the first element is the whole SV
            sv_region_df = sv_region_bed.to_dataframe()
            sv_region_df['name'] = 'sv'
            elements_coords_df = pd.concat([sv_region_df, elements_coords_df])
            gene_indicator = [0]+gene_indicator
        if force_region: #force sv region, useful in full mode
            elements_coords_df = sv_region_bed.to_dataframe()
            elements_coords_df['name'] = 'sv'
            gene_indicator = [0]
        #get feature array
        _, feature_array, _, element_name = read_features(elements_coords_df, feature_files=feature_files)
        if scaler_file is not None:
            feature_array = scale_features(feature_array, scaler_file=scaler_file)
        if sv_type=='deletion':
            feature_array = np.concatenate((feature_array, np.zeros(list(feature_array.shape)[:-1]+[1])), axis=-1)
        elif sv_type=='duplication':
            feature_array = np.concatenate((feature_array, np.ones(list(feature_array.shape)[:-1]+[1])), axis=-1)
        print('saving ' + sv_name)
        np.savez(target_file_path, element_names=element_name, feature_array=feature_array,
                 gene_indicator=gene_indicator, gene_names=gene_names)


def save_sv_features_global(sv_df, target_folder,feature_files,
                     elements_path,tad_path,scaler_file):
    if tad_path is not None:
        _, _, tad = utility.read_bed(tad_path, N=None, parse=False)
        tad_r = tad.loc[tad.iloc[:, 3] == "0", :2]
        tad_b = tad.loc[tad.iloc[:, 3] == "1", :2]
    else:
        tad_r = None
        tad_b = None
    elements_bb = pyBigWig.open(elements_path, "r")

    for i in range(sv_df.shape[0]):
        sv_tri = sv_df.iloc[i, :3].tolist()
        sv_type = sv_df.iloc[i, 4]
        sv_name = sv_df.iloc[i, 3]
        sv_bed = utility.tri_to_bed(sv_tri)

        assert sv_type == 'deletion' or sv_type == 'duplication', f'{sv_type} is not supported'

        target_file_path = os.path.join(target_folder, sv_name + '.npz')

        # get elements
        if tad_path is not None:
            tad_region_tri = superset_tad(sv_tri, tad_r, tad_b)
        else:
            sv_flank = sv_bed.flank(genome='hg38', b=1000000).to_dataframe()
            tad_region_tri = [sv_tri[0], int(min(sv_flank.start.min(),sv_tri[1])), int(max(sv_flank.end.max(),sv_tri[2]))]

        elements_coords_df, elements_coords_bed, _, gene_names = tad_elements(elements_bb, tad_region_tri)

        elements_start = elements_coords_bed.to_dataframe().start.min()
        elements_end = elements_coords_bed.to_dataframe().end.max()

        if sv_tri[1]-tad_region_tri[1]>1:
            left_bed = utility.tri_to_bed([elements_coords_bed[0].chrom, elements_start, sv_bed[0].start])
            df_left = elements_coords_bed.intersect(left_bed).to_dataframe()
        else:
            df_left = None

        if tad_region_tri[2]-sv_tri[2]>1:
            right_bed = utility.tri_to_bed([elements_coords_bed[0].chrom, sv_bed[0].end, elements_end])
            df_right = elements_coords_bed.intersect(right_bed).to_dataframe()
        else:
            df_right = None

        df_sv = elements_coords_bed.intersect(sv_bed).to_dataframe()

        # get feature array
        #----left
        if df_left is not None:
            _, feature_array_left, _, element_name_left = read_features(df_left, feature_files=feature_files)
            if scaler_file is not None:
                feature_array_left = scale_features(feature_array_left, scaler_file=scaler_file)
            feature_array_left = np.concatenate(
                (feature_array_left, 0.5+np.zeros((feature_array_left.shape[0], feature_array_left.shape[1], 1))), axis=-1)
        else:
            feature_array_left = None
            element_name_left = None
        #----sv
        _, feature_array_sv, _, element_name_sv = read_features(df_sv, feature_files=feature_files)
        element_name_sv_suffix = [e+'_sv' for e in element_name_sv]
        if scaler_file is not None:
            feature_array_sv = scale_features(feature_array_sv, scaler_file=scaler_file)
        if sv_type == 'deletion':
            feature_array_sv = np.concatenate(
            (feature_array_sv, np.zeros((feature_array_sv.shape[0], feature_array_sv.shape[1], 1))), axis=-1)
        else:
            feature_array_sv = np.concatenate(
                (feature_array_sv, np.ones((feature_array_sv.shape[0], feature_array_sv.shape[1], 1))), axis=-1)

        #----right
        if df_right is not None:
            _, feature_array_right, _, element_name_right= read_features(df_right, feature_files=feature_files)
            if scaler_file is not None:
                feature_array_right = scale_features(feature_array_right, scaler_file=scaler_file)
            feature_array_right = np.concatenate(
                (feature_array_right, 0.5+np.zeros((feature_array_right.shape[0], feature_array_right.shape[1], 1))), axis=-1)
        else:
            feature_array_right = None
            element_name_right = None

        array_list = [f for f in [feature_array_left, feature_array_sv, feature_array_right] if f is not None]
        feature_array = np.concatenate(array_list, axis=0)
        element_name_list = [element_name_left,element_name_sv_suffix,element_name_right]
        element_name_list = [ele for ele in element_name_list if ele is not None]
        element_name = utility.unpack_list(element_name_list)

        element_name_list_ = [element_name_left, element_name_sv, element_name_right]
        element_name_list_ = [ele for ele in element_name_list_ if ele is not None]
        element_name_ = utility.unpack_list(element_name_list_)

        gene_indicator = [1 if e in gene_names else 0 for e in element_name_]#intron is accounted as genes

        print('saving ' + sv_name)
        np.savez(target_file_path, element_names=element_name, feature_array=feature_array,
                 gene_indicator=gene_indicator, gene_names=gene_names)

def save_sv_features_del_dup(sv_df,target_folder,feature_files, elements_path,tad_path,scaler_file,annotation_path):
    _, sv_df_coding, sv_df_noncoding = utility.annot_sv(sv_df, annotation_path)
    if sv_df_coding.shape[0]>0:
        save_sv_features_local(sv_df_coding, target_folder,feature_files, elements_path,scaler_file)
    if sv_df_noncoding.shape[0]>0:
        save_sv_features_global(sv_df_noncoding, target_folder, feature_files, elements_path, tad_path,scaler_file)


def save_sv_features_ins_inv(sv_df, target_folder,feature_files, annotation_path,elements_path, tad_path,scaler_file):
    for i in range(sv_df.shape[0]):
        CHR, START, END, ID, sv_type = sv_df.iloc[i, 0],int(sv_df.iloc[i, 1]),int(sv_df.iloc[i, 2]),sv_df.iloc[i, 3],sv_df.iloc[i, 4]
        assert sv_type == 'inversion' or sv_type == 'insertion', f'{sv_type} is not supported'
        sv = of.sv_transformation(CHR, START, END, sv_type, ID, elements_path, annotation_path, None, None,full_mode=False)
        sv=sv[['CHR','START','END','ID','SVTYPE']]
        if sv.shape[0]==1:
            save_sv_features_del_dup(sv, target_folder, feature_files, elements_path, tad_path, scaler_file,annotation_path)
        else:
            sv['ID'] = [str(id)+str(i) for i,id in enumerate(sv['ID'].tolist())]
            target_folder_sv = os.path.join(target_folder,ID)
            if not os.path.isdir(target_folder_sv):
                os.makedirs(target_folder_sv)
            save_sv_features_del_dup(sv,target_folder_sv,feature_files, elements_path,tad_path,scaler_file,annotation_path)

def save_sv_features_auto(sv_df, target_folder,feature_files, annotation_path,elements_path, tad_path,scaler_file):
    sv_df_del_dup = sv_df[sv_df.iloc[:,4].isin(['deletion','duplication'])].reset_index(drop=True)
    sv_df_ins_inv = sv_df[sv_df.iloc[:,4].isin(['insertion', 'inversion'])].reset_index(drop=True)
    if sv_df_del_dup.shape[0]>0:
        save_sv_features_del_dup(sv_df_del_dup, target_folder, feature_files, elements_path, tad_path, scaler_file, annotation_path)
    if sv_df_ins_inv.shape[0] > 0:
        save_sv_features_ins_inv(sv_df_ins_inv, target_folder, feature_files, annotation_path,elements_path, tad_path, scaler_file)


def sv_to_tad(sv_df,tad_path):
    # input sv_df
    # output tad_df
    _, _, tad = utility.read_bed(tad_path, N=None, parse=False)
    tad_r = tad.loc[tad.iloc[:, 3] == "0", :2]
    tad_b = tad.loc[tad.iloc[:, 3] == "1", :2]
    ID = []
    CHR=[]
    START=[]
    END=[]
    for i in range(sv_df.shape[0]):
        sv_tri = sv_df.iloc[i, :3].tolist()
        sv_name = sv_df.iloc[i, 3]
        tad_region_tri = superset_tad(sv_tri, tad_r, tad_b)
        ID.append(sv_name)
        CHR.append(tad_region_tri[0])
        START.append(tad_region_tri[1])
        END.append(tad_region_tri[2])
    df = pd.DataFrame({'ID':ID,'CHR':CHR,'TAD_START':START,'TAD_END':END})
    return df
