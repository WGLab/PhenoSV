import pyBigWig
import gzip
import os
import pandas as pd
import numpy as np
import math
from itertools import chain
from pybedtools import BedTool

########################################
##############general###################
########################################

def unpack_list(x):
    return list(chain(*x))


def read_top(path, N):
    header = []
    lines = []
    with gzip.open(path, 'rt') as file:
        for i in range(N):
            line = next(file).strip()
            if not line.startswith('#'):
                line = line.split('\t')
                lines.append(line)
            else:
                header.append(line)
    return header, lines

#parse INFO into a dict
def parse_info(x: str,sep=';',inner_sep = '=')->dict:
    '''
    parse INFO into a dict
    :param x: a string
    :param sep: between different elements
    :param inner_sep: between key and value
    :return: a dict with parsed info
    '''
    x=x.split(sep)
    x = dict([(i.split('=')[0],i.split(inner_sep)[1]) for i in x])
    return x

#lines into dataframe
def lines2df(lines: list, sep=';',inner_sep = '=', parse = True):
    '''
    :param lines: a list to transform into dataframe
    :param parse: whether to parse the last column
    :param sep: between different elements
    :param inner_sep: between key and value
    :return: dataframe
    '''

    df = pd.DataFrame(lines)
    if parse:
        info = list(df.iloc[:, -1])
        info = [parse_info(x, sep, inner_sep) for x in info]
        df2 = pd.DataFrame(info)
        df1 = df.iloc[: , :-1]
        df = pd.concat([df1, df2], axis=1)
    return df

#one hot encoding
def one_hot(input: int, length: int)->list:
    '''
    one hot encoding an int
    :param input: int
    :param length: total number of categories
    '''
    out = np.zeros(length).astype(int).tolist()
    if input>0:
        out[input-1]=1
    return out

def one_hot_list(input: list, length: int):
    '''
    one hot encoding a list of int
    :param input: list of int
    :param length: total number of categories
    :return array of encoding, each row is an element from input list
    '''
    out = [one_hot(i,length) for i in input]
    out = np.array(out)
    return out


def segment_stat(segment_list: list, start: int, end: int, categories: int, stat='mean'):
    '''
    given a start and an end point of a segment, calculate max or mean from segment list
    :param: segment_list: output of a query from bigbed or bed. must cover all the range
    :param start: list of int
    :param end: list of int
    :param categories: list of int, set to 0 or 1 if not one hot encoding
    :return list with length of categories
    '''
    start_list = []
    end_list = []
    value_list = []
    for segment in segment_list:
        start_list.append(int(segment[0]))
        end_list.append(int(segment[1]))
        value_list.append(int(segment[2]))
    start_list[0] = start
    end_list[-1] = end
    segment_length = np.array(end_list) - np.array(start_list) #1 * bins
    if categories>1:
        value = one_hot_list(value_list, categories) # 1 * categories--np array
    else:
        value = np.expand_dims(np.array(value_list), 1)
    out = [0]
    if stat=='max':
        out = np.max(value, axis=0).tolist() #1 * categories
    elif stat=='mean':
        out = (np.matmul(segment_length, value) / (end - start)).tolist()
    elif stat=='sum':
        out = np.matmul(segment_length, value).tolist()
    else:
        print('stat error, try max, mean or sum')
    return out

def percentile_rank(reference, features):
    features_norm=features.copy()
    for i in range(reference.shape[1]):
        ref=reference[:,i]
        features_norm[:,i] = (ref<features[:,i][:,None]).mean(axis=1)
    return features_norm

def list_file(rootfolder,file_format='.txt',wholepath= True):
    '''
    used to list all file paths of specific type under the root folder
    :param rootfolder: root folder containing svs files
    :return: list of svs paths
    '''
    path = []
    for root, directories, files in os.walk(rootfolder, topdown=False):
        for file in files:
            file_path=os.path.join(root, file)
            if file_format in file_path:
                path.append(file_path)
    if wholepath:
        path = [os.path.join(rootfolder, i) for i in path]
    return path


def tri_to_bed(tri):
    tri[1] = int(tri[1])
    tri[2] = int(tri[2])
    region_bed = BedTool(' '.join([str(i) for i in tri]), from_string=True)
    return region_bed

def tri_to_string(tri):
    tri[1] = int(tri[1])
    tri[2] = int(tri[2])
    region_str = ' '.join([str(i) for i in tri])
    return region_str

########################################
##############bigwig ###################
########################################
def chunk_list(l,nbins):
    step = math.ceil(len(l) / nbins)
    chunks = [l[x:x + step] for x in range(0, len(l), step)]
    return chunks


def summary_bw(bw_path: str)->dict:
    '''
    summary a bigwig file with header and chromosome information
    :param bw_path: path of file
    :return: dict summary
    '''
    bw = pyBigWig.open(bw_path, "r")
    summary = {'header':bw.header(),'chroms':bw.chroms()}
    return summary


def query_bw(bw_path: str, chrom: str, start: int, end: int, nbins: int, stat='max', onlycoverage=1,
             binlen=None) -> list:
    OUT=[0]*nbins
    if os.path.isfile(bw_path):
        bw = pyBigWig.open(bw_path, "r")
        if binlen is not None:
            n_bin = math.ceil((end-start)/binlen)
            if n_bin>1:
                start_pos = np.linspace(start,int(end-binlen),n_bin,dtype=int)
                end_pos = [s+binlen for s in start_pos]
            else:
                start_pos=[start]
                end_pos=[end]
            bins = list(zip(start_pos, end_pos))
            nbins = 1
        else:
            bins = [(start, end)]
        #if no chrX
        if chrom in list(bw.chroms().keys()):
            OUT = []
            for Bin in bins:
                if stat == 'q95':
                    l = bw.values(chrom, Bin[0], Bin[1])
                    l_chunks = chunk_list(l, nbins)
                    outlist = [np.nanquantile(l, 0.95) for l in l_chunks]
                    outlist = [0 if math.isnan(i) else i for i in outlist]
                elif stat == 'mean' and onlycoverage == 0:
                    outlist = bw.stats(chrom, Bin[0], Bin[1], type='sum', nBins=nbins, exact=True)
                    bin_length = round((end - start) / nbins)
                    outlist = [i / bin_length if i is not None else 0 for i in outlist]
                else:
                    outlist = bw.stats(chrom, Bin[0], Bin[1], type=stat, nBins=nbins, exact=True)
                    outlist = [i if i is not None else 0 for i in outlist]
                OUT.append(outlist)
                OUT = np.hstack(OUT).tolist()
    return OUT


########################################
##############bigbed ###################
########################################

def query_bb(bb_path: str, chrom: str ,start: int,end: int, nbins: int, stat = 'max', categories=100,binlen=None):
    bb = pyBigWig.open(bb_path, "r")
    if binlen is not None:
        n_bin = math.ceil((end - start) / binlen)
        if n_bin>1:
            start_pos = np.linspace(start, int(end - binlen), n_bin, dtype=int)
            end_pos = [s + binlen for s in start_pos]
        else:
            start_pos=[start]
            end_pos=[end]
        bins = list(zip(start_pos, end_pos))
    else:
    #generate bin-wise start and end
        start_pos = [start]+[math.floor(start+(i+1)*(end-start)/nbins) for i in range(nbins)]
        start_pos = start_pos[:-1]
        end_pos = start_pos[1:]+[end]
        bins = list(zip(start_pos, end_pos))

    out_bins = []
    for bin_ in bins:
        segment_list = bb.entries(chrom, bin_[0], bin_[1])
        if segment_list is None:
            segment_list = [(bin_[0], bin_[1],'0')]
        segment_stat_ = segment_stat(segment_list, start=bin_[0], end=bin_[1], categories= categories,
                                     stat= stat) #1* categories
        out_bins.append(segment_stat_)
    out_bins = np.array(out_bins) #bin * state
    return out_bins




########################################
##############gff3   ###################
########################################

def read_gff3_zip(path: str, N: int):
    '''
    read gff3 file into a dataframe
    :param path: path of gff3 file
    :param N: only read top N lines
    :return: dadaframe
    '''
    header = []
    lines = []
    with gzip.open(path, 'rt') as file:
        if N is not None:
            for i in range(N):
                line = next(file).strip()
                if not line.startswith('#'):
                    line = line.split('\t')
                    lines.append(line)
                else:
                    header.append(line)
        else:
            for line in file:
                if not line.startswith('#'):
                    line = [x for x in line.split('\t')]
                    lines.append(line)
        df = lines2df(lines, parse=True)

    return header, lines, df


def separate_gff3(input_file_path: str, output_folder_path: str):
    #create files
    #gene annotations
    fgene = open(os.path.join(output_folder_path, 'gene_annotation') + '.bed', 'w')
    fgene.write('#chrom\tchromStart\tchromEnd\tname\n')
    fgene.close()
    #transcript annotations
    ftrans = open(os.path.join(output_folder_path, 'transcript_annotation') + '.bed', 'w')
    ftrans.write('#chrom\tchromStart\tchromEnd\tname\n')
    ftrans.close()
    #exon annotations
    fexon = open(os.path.join(output_folder_path, 'exon_annotation') + '.bed', 'w')
    fexon.write('#chrom\tchromStart\tchromEnd\tname\n')
    fexon.close()
    #utr annotations
    futr3 = open(os.path.join(output_folder_path, 'utr3_annotation') + '.bed', 'w')
    futr3.write('#chrom\tchromStart\tchromEnd\tname\n')
    futr3.close()
    futr5 = open(os.path.join(output_folder_path, 'utr5_annotation') + '.bed', 'w')
    futr5.write('#chrom\tchromStart\tchromEnd\tname\n')
    futr5.close()
    with gzip.open(input_file_path, 'rt') as file:
        for line in file:
            if not line.startswith('#'):
                line = [x for x in line.split('\t')]
                ##save gene
                if line[2]=='gene':
                    info = parse_info(line[8])
                    if info['gene_type'] == 'protein_coding':
                        fgene = open(os.path.join(output_folder_path, 'gene_annotation')+'.bed','a')
                        fgene.write('{}\t{}\t{}\t{}\n'.format(line[0], int(line[3]) - 1, line[4],
                                            parse_info(line[8])['gene_name']))
                        fgene.close()
                ##save transcript---basic transcript
                if line[2]=='transcript':
                    info = parse_info(line[8])
                    if info['transcript_type']=='protein_coding' and info['level']in('1','2') and 'basic' in info['tag']:
                        ftrans = open(os.path.join(output_folder_path, 'transcript_annotation')+'.bed','a')
                        ftrans.write('{}\t{}\t{}\t{}\n'.format(line[0], int(line[3]) - 1, line[4],info['transcript_name']))
                        ftrans.close()
                ##save exon
                if line[2]=='exon':
                    info = parse_info(line[8])
                    if info['transcript_type']=='protein_coding' and info['level']in('1','2') and 'basic' in info['tag']:
                        fexon = open(os.path.join(output_folder_path, 'exon_annotation')+'.bed','a')
                        fexon.write('{}\t{}\t{}\t{}\n'.format(line[0], int(line[3]) - 1, line[4],info['transcript_name']))
                        fexon.close()
                ##save 5utr
                if line[2] == 'five_prime_UTR':
                    info = parse_info(line[8])
                    if info['transcript_type'] == 'protein_coding' and info['level'] in ('1', '2') and 'basic' in info[
                        'tag']:
                        futr5 = open(os.path.join(output_folder_path, 'utr5_annotation') + '.bed', 'a')
                        futr5.write(
                            '{}\t{}\t{}\t{}\n'.format(line[0], int(line[3]) - 1, line[4], info['transcript_name']))
                        futr5.close()
                if line[2] == 'three_prime_UTR':
                    info = parse_info(line[8])
                    if info['transcript_type'] == 'protein_coding' and info['level'] in ('1', '2') and 'basic' in info[
                        'tag']:
                        futr3 = open(os.path.join(output_folder_path, 'utr3_annotation') + '.bed', 'a')
                        futr3.write(
                            '{}\t{}\t{}\t{}\n'.format(line[0], int(line[3]) - 1, line[4], info['transcript_name']))
                        futr3.close()


########################################
##############bed    ###################
########################################

def read_bed(path: str, N: int, parse= True):
    '''
    read bed file into a dataframe
    :param path: path of gff3 file
    :param N: only read top N lines
    :return: dadaframe
    '''
    header = []
    lines = []
    with open(path, 'rt') as file:
        if N is not None:
            for i in range(N):
                line = next(file).strip()
                if not line.startswith('#'):
                    line = line.split('\t')
                    lines.append(line)
                else:
                    header.append(line)
        else:
            for line in file:
                if not line.startswith('#'):
                    line = line.strip()
                    line = [x for x in line.split('\t')]
                    lines.append(line)
        df = lines2df(lines, parse=parse)

    return header, lines, df


def read_bed_zip(path: str, N: int, parse = True):
    '''
    read zipped bed file into a dataframe
    :param path: path of gff3 file
    :param N: only read top N lines
    :return: dadaframe
    '''
    header = []
    lines = []
    with gzip.open(path, 'rt') as file:
        if N is not None:
            for i in range(N):
                line = next(file).strip()
                if not line.startswith('#'):
                    line = line.split('\t')
                    lines.append(line)
                else:
                    header.append(line)
        else:
            for line in file:
                if not line.startswith('#'):
                    line = line.strip()
                    line = [x for x in line.split('\t')]
                    lines.append(line)
        df = lines2df(lines, parse=parse)

    return header, lines, df



def sample_by_af(background_svpath, n=100, weight_name='AF'):
    if 'csv' in background_svpath:
        sv = pd.read_csv(background_svpath)
    else:
        sv = pd.read_csv(background_svpath,sep='\t')

    weights=sv[weight_name]
    sv_sample = sv.sample(n=n, replace=False, weights=weights, ignore_index=True)
    return sv_sample


def simulation_background(background_svpath, n=25000, drop_common = False):
    sv = sample_by_af(background_svpath, n, weight_name='AF')
    common_id = sv[sv.AF > 0.01].iloc[:,3].tolist()
    sv_bed = BedTool.from_dataframe(sv.iloc[:, [0, 1, 2, 6, 3]])
    sv = sv_bed.sort().cluster().to_dataframe()
    sv = sv.groupby('strand').apply(lambda x: x.sample(1)).reset_index(drop=True)
    sv = sv.drop(['strand'],axis=1)
    sv.columns=['CHR','START','END','SVTYPE','ID']
    # drop common
    if drop_common:
        sv = sv[sv.ID.isin(common_id)==False].reset_index(drop=True)
    return sv


def annot_sv(sv_df,annotation_path):
    if isinstance(annotation_path, str):
        annot_bed = BedTool(annotation_path)
    else:
        annot_bed = annotation_path
    sv_bed = BedTool.from_dataframe(sv_df.iloc[:, :4])
    sv_coding = sv_bed.intersect(annot_bed, u=True).to_dataframe()
    if sv_coding.shape[0]==0:
        sv_df['annotation']=['noncoding']*sv_df.shape[0]
    else:
        sv_df['annotation']=['coding' if s in sv_coding['name'].tolist() else 'noncoding' for s in sv_df['ID'].tolist()]
    sv_coding = sv_df[sv_df['annotation'] == 'coding'].reset_index(drop=True)
    sv_noncoding = sv_df[sv_df['annotation'] == 'noncoding'].reset_index(drop=True)
    return sv_df, sv_coding, sv_noncoding

def annot_sv_single(chr,start,end,annotation_path):
    if isinstance(annotation_path, str):
        annot_bed = BedTool(annotation_path)
    else:
        annot_bed = annotation_path
    sv_bed = tri_to_bed([chr,start,end])
    sv_coding=sv_bed.intersect(annot_bed, u=True).to_dataframe()
    if sv_coding.shape[0]==0:
        sv_type='noncoding'
    else:
        sv_type='coding'
    return sv_type


def liftover(chromosome,position,converter):
    conv = converter[chromosome][position]
    if len(conv)==0:
        position = -1
    else:
        position = conv[0][1]
    return position