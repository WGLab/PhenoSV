# PhenoSV: Interpretable phenotype-aware model for the prioritization of genes affected by structural variants.
[![DOI](https://zenodo.org/badge/628754962.svg)](https://zenodo.org/doi/10.5281/zenodo.10028740)

## Background
Structural variants (SVs) represent a major source of genetic variation and may be associated with phenotypic diversity and disease susceptibility. Recent advancements in long-read sequencing have revolutionized the field of SV detection, enabling the discovery of over 20,000 SVs per human genome. However, identifying and prioritizing disease-related SVs and assessing their functional impacts on individual genes remains challenging, especially for noncoding SVs. 

PhenoSV is a phenotype-aware machine-learning model to predict pathogenicity of all types of structural variants (SVs) that disrupt either coding or noncoding genome regions, including deletions, duplications, insertions, inversions, and translocations. PhenoSV segments SVs and annotates each segment using hundreds of genomic features, then adopts a transformer-based architecture to predict functional impacts of SVs under a multiple-instance learning framework. When phenotype information is available, PhenoSV further utilizes gene-phenotype associations to prioritize disease-related SVs. 

## Web server
For SVs that affect less than 30 protein-coding genes (10 protein-coding genes each for batch predictions), we provide a web server at https://phenosv.wglab.org for easy applications of PhenoSV. If you want to score SVs that affect more than 30 genes or make batch predictions, please install PhenoSV and run offline. 

## Installation

### Step1: install sources 
To avoid package version conflicts, we strongly recommand to use conda to set up the environment. If you don't have conda installed, you can run codes below in linux to install.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh
```

After installing conda, PhenoSV sources can be downloaded:

```
git clone https://github.com/WGLab/PhenoSV.git
cd PhenoSV
conda env create --name phenosv --file phenosv.yml
conda activate phenosv
```

### Step2: download and set up required files
Some files are required by PhenoSV, including genomic feature files, the pre-trained PhenoSV model, Phen2Gene knowledgebases, etc. All of these files required by the full version of PhenoSV have been packed together and can be downloaded using codes below. The packed file takes about 160G storage, please make sure to store these files under a directory with enough space. This directory can be different from the one used to save PhenoSV source codes. Due to the large size of feature files, this step might take some time. Simply run codes below to set up everything. 

```
bash setup.sh /path/to/folder
```

We also offered a light-weight version of PhenoSV as a highly efficient alternative of PhenoSV. PhenoSV-light consists of only 42 features with much improved annotation efficiency without compromising predictive accuracy except for translocations. The packed file takes about 50G storage. Run the codes below to set up PhenoSV-light. 

```
bash setup.sh /path/to/folder 'light'
```

Note that, users who downloaded the full set of required files using can excute both PhenoSV and PhenoSV-light. Users who downloaded the light version files can only excute PhenoSV-light.

If users need to change the path for storing required data files and would like to make changes for the config file, below codes can be excuted.

```
bash update_config.sh /path/to/newfolder 
```


### Step3: install PhenoSV as a python package (optional)

This step is not required. If you want to integrate PhenoSV into your own python scripts, you can install PhenoSV as a python package following the steps above.

```
pip install -e .
```

## Run PhenoSV in linux

In linux, PhenoSV can be used to: 
- score a single SV with or without prior phenotype knowledge 
- score a list of SVs in .bed, .csv, or .bedpe format with or without prior phenotype knowledge. 

Type `python3 phenosv/model/phenosv.py -h` to see all options. 

```
options:
  -h, --help            show this help message and exit
  --genome GENOME       choose genome build between hg38 (default) and hg19
  --alpha ALPHA         A positive value with larger value representing more contribution of phenotype information in refining PhenoSV
                        scores. Default is 1
  --inference INFERENCE
                        leave it blank (default) if only considering direct impacts of coding SVs. Set to `full` if inferring both
                        direct and indirect impacts of coding SVs
  --model MODEL         choose between PhenoSV (default) and PhenoSV-light
  --c C                 chromosome, e.g. chr1
  --s S                 start, e.g. 2378909
  --e E                 end, e.g. 2379909
  --c2 C2               chromosome2, e.g. chr1, only for translocation
  --s2 S2               start2, e.g. 2378909, only for translocation
  --strand1 STRAND1     strand1, + or - , only for translocation
  --strand2 STRAND2     strand2, + or - , only for translocation
  --svtype SVTYPE       deletion, duplication, insertion, inversion, translocation
  --noncoding NONCODING
                        inference mode, choose from distance and tad
  --HPO HPO             HPO terms should in the format of HP:digits, e.g., HP:0000707, separated by semicolons, commas, or
                        spaces.
  --sv_file SV_FILE     path to SV file (.csv, .bed, .bedpe)
  --target_folder TARGET_FOLDER
                        enter the folder path to save PhenoSV results, leave it blank if you only want to print out the results
  --target_file_name TARGET_FILE_NAME
                        enter the file name to save PhenoSV results
```


### Score a single SV

The running time of PhenoSV to score a single SV depends on the number of genes it impacted. For the examples below, PhenoSV is expected to generate results within a few seconds.

#### deletion, duplication, insertion, inversion
##### Example1
You can use the following codes to score a single SV (deletion, duplication, insertion, inversion) easily by providing the SV location and type. The arguments required are: --c: chromosome, --s: start position, --e: end position (can be ignored by insertions), --svtype: types of SV.

```
python3 phenosv/model/phenosv.py --c chr6 --s 156994830 --e 157006982 --svtype 'deletion'
```

##### Example2
Since this example SV is a noncoding SV, PhenoSV's default setting is to consider genes within 1Mbp upstream and downstream impacted. PhenoSV can also consider genes based on consensus TAD annotation by setting `--noncoding` argument as 'tad'. Prior phenotype information can be added using `--HPO` argument. Here is an example:

```
python3 phenosv/model/phenosv.py --c chr6 --s 156994830 --e 157006982 --svtype 'deletion' --noncoding 'tad' --HPO 'HP:0000707,HP:0007598'
```

PhenoSV will output results below. Without considering phenotype information, PhenoSV predicts the SV-level pathogenicity as 0.65. The gene-level pathogenicity scores are 0.82 for ARID1B by disrupting its introns, 0.05 for NOX3, and 0.53 for TFB1M by indirectly altering their regulatory elements. After adding phenotype information, PhenoSV scores are 0.65 for the whole SV and 0.82 for ARID1B gene.

```
  Elements  Pathogenicity           Type  Phen2Gene   PhenoSV
0       SV       0.664912  Non-coding SV   0.999126  0.664331
1   ARID1B       0.823556       Intronic   0.999126  0.822836
2     NOX3       0.045648     Regulatory   0.837460  0.038229
3    TFB1M       0.533431     Regulatory   0.544762  0.290593

```

##### Example3
Another example as shown in the paper, we investigated the SV that indirectly impacting SOX9 (Kurth et al. 2009, GRCh38, chr17: 70134929-71339950, duplication). This is a coding SV impacting exons of gene KCNJ16 and KCNJ2, as seen below. The model only predict direct impacts of coding SVs on genes within the SV region by default, and the pathogenicity score is 0.07, very likely to be benign.

```
python3 phenosv/model/phenosv.py --c chr17 --s 70134929 --e 71339950 --svtype 'duplication'
```

```
  Elements  Pathogenicity           Type
0       SV       0.066281      Coding SV
1   KCNJ16       0.016484         Exonic
2    KCNJ2       0.088711         Exonic

```

We can set add argument of `--inference 'full'` to infer both direct and indirect impacts of coding SVs. Here the model predict the SV pathogenicity score as 0.69 throught impacting genes indirectly, whereas the gene-level pathogenicity scores for MAP2K6 and SOX9 are 0.59 and 0.61 respectively. 

```
python3 phenosv/model/phenosv.py --c chr17 --s 70134929 --e 71339950 --svtype 'duplication' --inference 'full' --noncoding 'tad'
```

```
  Elements  Pathogenicity                Type
0       SV       0.066281           Coding SV
1   KCNJ16       0.016484              Exonic
2    KCNJ2       0.088711              Exonic
3       SV       0.688360  Coding SV indirect
4   MAP2K6       0.594582          Regulatory
5     SOX9       0.610116          Regulatory

```


##### Example4

To run the PhenoSV-light model, simply add --model 'PhenoSV-light' as shown below.

```
python3 phenosv/model/phenosv.py --c chr6 --s 156994830 --e 157006982 --svtype 'deletion' --model 'PhenoSV-light'
```

#### translocation

PhenoSV can also be used to interpret translocations. The arguments required are: --c: 5' chromosome, --s: 5' breakpoint, --c2: 3' chromosome, --s2: 3' breakpoint, --svtype: types of SV (translocation); --strand1:  5' strand and --strand2:  3' strand are optional with '+' as default.

```
python3 phenosv/model/phenosv.py --c chr6 --s 156994830 --strand1 '+' --c2 chr7 --s2 156994830 --strand2 '+' --svtype 'translocation'
```

PhenoSV will output results below. The SV-level pathogenicity is 0.98, generating a fusion ARID1B-MNX1 gene with PhenoSV scores being 0.98 and 0.95 for ARID1B and MNX1, respectively.

```
  Elements  Pathogenicity       Type                             ID
0       SV       0.975636  Coding SV  chr6:156994830-chr7:156994830
1   ARID1B       0.975636     Exonic  chr6:156994830-chr7:156994830
2     MNX1       0.947925     Exonic  chr6:156994830-chr7:156994830
```

### Score multiple SVs

PhenoSV accepts csv, bed, and bedpe files as input to score multiple SVs. Some examples are provided at `data/`. csv and bed files can be used to score deletion, duplication, inversion, and insertion. bedpe files can be used to score translocations. Fields of csv and bed files should be: chromosome, start, end, ID, svtype, HPO (optional). Fields of bedpe files should be: chromosome1, start1, end1, chromosome2, start2, end2, strand1, strand2, ID. start1 and start2 will be treated as breakpoints, whereas end1 and end2 will not be used by PhenoSV. Note that, if input files do not have HPO terms, you can use the `--HPO` argument to add phenotype information by treating the same HPO terms for all SVs in the file. If HPO terms are present in input files, the `--HPO` argument will be ignored.

To score multiple SVs using a single process, run: 

```
python3 phenosv/model/phenosv.py --sv_file data/sampledata.bed --target_folder data/ --target_file_name sample_bed_out
```

You can also score multiple SVs in parallel to speed up. Below is an example of running PhenoSV with 4 processes in parallel (set up using `--workers` argument). Leave the HPO terms blank if they are already in the input file or you don't have prior phenotype information. 

```
bash phenosv/model/phenosv.sh --sv_file 'path/to/sv/data.csv' --target_folder 'folder/path/to/store/results' --workers 4 --HPO 'HP:0000707,HP:0007598' --model PhenoSV-light
```



## Run PhenoSV in Python

You can run PhenoSV in Python and the output will be a pandas dataframe with the SV-level and the gene-level predictions for a given SV.

```
#import packages
import os
import phenosv
from phenosv.model.phenosv import init as init
import phenosv.model.operation_function as of

#get configurations, use the light argument for choosing between PhenoSV and PhenoSV-light
config_path = os.path.join(os.path.dirname(phenosv.__file__), '..', 'lib', 'fpath.config')
configs, ckpt = init(config_path, ckpt = True, light = False)

# set 'tad_path' as None to consider genes within 1Mbp uptream and downstream a noncoding SV.
# do not run this line if you want to use TAD annotations to interpret noncoding SVs
configs['tad_path']=None

# users can also specify tissue-specific TAD annotations based on research goals
# simply assign TAD annotation path using configs['tad_path']='path/to/tad_annotation.bed'

#load model
model = of.prepare_model(ckpt)

#to interpret a single SV that is not translocation
of.phenosv(CHR='chr6', START=156994830, END=157006982, svtype='deletion', model=model, HPO=None, **configs)

#to interpret a single SV that is translocation
of.phenosv(CHR='chr6', START=156994830,END=None,svtype='translocation', model=model, HPO=None,
           CHR2='chr11', START2=111728347, strand1='+',strand2='+',**configs)

#liftover if using hg19 build
from liftover import get_lifter
import phenosv.utilities.utility as u
converter = get_lifter('hg19', 'hg38')

#SV in hg19
CHR, START, END ='chr6',157315964, 157328116 
#liftover to hg38
START = u.liftover(CHR, START, converter)
END = u.liftover(CHR, END, converter)
of.phenosv(CHR=CHR, START=START, END=END, svtype='deletion', model=model, HPO=None, **configs)

```
## User defined TAD annotations

PhenoSV relies on pre-determined sets of candidate genes when interpreting the impacts of noncoding SVs, either by distance or TAD annotations. We already included an aggregated version of TAD annotation file with PhenoSV. This annotation file is tissue-unspecific. Users can access to this file under the `data` folder at pre-assigned path (saved in lib/fpath.config). The file name is `tad_w_boundary_08.bed`

Since TAD annotations are tissue specific, users can also use their own TAD annotations depending on different study goals. The annotation file should be in bed format. Run codes below to assign different annotation files. 

```
python3 phenosv/model/phenosv.py --c chr6 --s 156994830 --e 157006982 --svtype 'deletion' --noncoding 'path/to/tad_annotation.bed'
```

Here is an example showing the format of the TAD annotation file. Four columns are 'CHR', 'START','END', and 'Indicators' (0 means TAD domain, 1 means TAD boundaries).

```
chr1    0       724620  0
chr1    724620  764620  1
chr1    764620  3423436 0
chr1    3423436 3463436 1
chr1    3463436 3783436 0
chr1    3783436 3823436 1
```


## Annotate SVs

Using the above codes, PhenoSV annotates SVs with hundreds of genomic features on the fly and then feeds into the pre-trained model to make pathogenicity calls. Users can also save SV annotations forehead and make predictions afterward using the codes below. 

```
#Annotate SVs using a single CPU core
python3 phenosv/model/annotation.py --sv_file data/sampledata.csv --target data/

#Annotate SVs using multiple CPU cores in parallel (4 cores in the example)
bash annotation.sh data/sampledata.csv path/to/output/folder/ 4
```

## Archived datasets

We deposited simulated patients' SV profiles used in manuscript for prioritizations. Several things to notice:
- Each file corresponds to one patient's SV profile after filtering out all common SVs
- Within each SV profile, the first row is the real disease-associated SV, the rest SVs are noise rare SVs. 
- We hided the coordinates of all SVs from DECIPHER, but one can query the information from https://www.deciphergenomics.org 
- The column of `Pathogenicity` represents general pathogenicity scores predicted by PhenoSV ($p_{sv}$), `Phen2Gene` represents gene-phenotype associations, and `PhenoSV` represents SV pathogenicity associated with given phenotype information ($p_{sv}^{pheno}$) when setting $\alpha=1$.

Use the codes below to download the simulation data. 

```
wget https://www.openbioinformatics.org/PhenoSV/prioritization_simulation_benchmark.tar.gz
```

Souce data of the paper used to generate all figures can be found  [here](https://github.com/WGLab/PhenoSV/tree/main/data)
