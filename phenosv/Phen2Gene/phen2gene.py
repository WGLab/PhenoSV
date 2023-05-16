import os
import sys
import io
import re
import pandas as pd
import numpy as np
from scipy import stats
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir,'..')
sys.path.insert(0, module_dir)

from Phen2Gene.prioritize import gene_prioritization
from Phen2Gene.weight_assignment import assign
from Phen2Gene.calculation import calc, calc_simple



def phen2gene(HPO,KBpath='/home/xu3/Phen2Gene/lib', scale_score=False):
    HPOlist = re.split(r';|,|\n|\s', HPO)
    HPOlist= [i.strip() for i in HPOlist if i.strip()]
    gene_dict, _, weight_model = results(KBpath, files=None, manuals=HPOlist, user_defineds=None,
                                              weight_model='sk', weight_only=False, output_path=None,
                                              output_file_name=None, gene_weight=None,
                                              cutoff=None, genelist=None, verbosity=True)
    if len(gene_dict)==0:
        df=None
    else:
        df = pd.DataFrame(gene_dict).transpose()
        df.columns = ['Gene','Score','status','status_','id']
        df = df[['Gene', 'Score']]
        df.Score = df.Score / np.max(df.Score)
        if scale_score:
            score = df.Score.tolist()
            score = stats.percentileofscore(score,score)/100
            df['Score'] = score.tolist()
    return df



def results(KBpath, files=None, manuals=None, user_defineds=None, weight_model=None, weight_only=False,
            output_path=None, output_file_name=None,  gene_weight=None, cutoff=None,
            genelist=None, verbosity=False):
    HPO_id = []
    # read what weighting model user input. Skewness is the default model.
    if (user_defineds != None):
        weight_model = 'd'
    else:
        if (weight_model == None or (
                weight_model.lower() != 'u' and weight_model.lower() != 's' and weight_model.lower() != 'ic' and weight_model.lower() != 'w')):
            weight_model = 'sk'
    # only for command line
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    # Collect manually input HPO ID
    if (manuals != None and weight_model != 'd'):
        for HP_item in manuals:
            # Simple check if HPO ID or not
            if (HP_item.startswith("HP:") and len(HP_item) == 10 and HP_item[3:].isnumeric()):
                HPO_id.append(HP_item.replace(':', '_'))
            else:
                if (verbosity):
                    print(HP_item, "is not a valid HPO ID,", sep=" ")
    # Extract HPO ID from input files
    if (files != None and weight_model != 'd'):
        for file_item in files:
            try:
                with open(file_item, "r") as one_file:
                    print("Reading", file_item, "...", "\n", sep=" ")
                    entire_data = one_file.read()
                    HP_data = entire_data.split("\n")
                    for HP in HP_data:
                        # Simply check if HPO ID or not
                        if (HP.startswith("HP:") and len(HP) == 10 and HP[3:].isnumeric()):
                            # Change HP id format from 'HP:nnnnnnn' to 'HP_nnnnnnn', since ':' is an illegal character in file names in MacOS ans Windows system
                            HPO_id.append(HP.replace(":", "_"))
            except FileNotFoundError:
                print("\n" + file_item + " not found!\n", file=sys.stderr)

    ## Create a dict to store weights of HPO terms
    hp_weight_dict = {}
    # If HPO weights are pre-defined by users
    if (weight_model == 'd'):
        if (user_defineds != None):
            for file_item in user_defineds:
                try:
                    with open(file_item, "r") as one_file:
                        entire_data = one_file.read()
                        entire_data = entire_data[:-1]
                        HP_weight_data = entire_data.split("\n")
                        for HP_weight in HP_weight_data:
                            hp_weight = HP_weight.split("\t")
                            # Change HP id format from 'HP:nnnnnnn' to 'HP_nnnnnnn', since ':' is an illegal character in file names in MacOS ans Windows system
                            hp = hp_weight[0]
                            hp = hp.replace(":", "_", 1)
                            hp_weight_dict[hp] = float(hp_weight[1])

                except FileNotFoundError:
                    print("\n" + file_item + " not found!\n", file=sys.stderr)
    # elif(weight_model == 'sk'):

    # HPO weights are determined by weighting models
    else:
        for hp in HPO_id:
            (weight, replaced_by) = assign(KBpath, hp, weight_model)
            if (weight > 0):
                if (replaced_by != None):
                    hp_weight_dict[replaced_by] = weight
                else:
                    hp_weight_dict[hp] = weight

    ### Only outputs HP id's weights
    if (weight_only):
        with open(output_path + output_file_name + ".HP_weights", "w+") as fw:
            fw.write("HP ID\tWeight\n")
            for hp in hp_weight_dict.keys():
                fw.write(hp + "\t" + str(hp_weight_dict[hp]) + "\n")
        print("Finished.")
        print("Output path: " + output_path + "\n")
        exit()
    ### down_weighting
    # Create a dict to store associated gene data
    if (weight_model.lower() == 's'):
        gene_dict = calc_simple(KBpath, hp_weight_dict, verbosity)
    else:
        gene_dict = calc(KBpath, hp_weight_dict, verbosity, gene_weight, cutoff)
    std_output = new_stdout.getvalue()
    # reset stdout
    sys.stdout = old_stdout
    ### if user inputs a candidate gene list, remove all genes but candidate genes
    if (genelist):
        with open(genelist) as f:
            genelist = [line.strip() for line in f]
        for key in genelist:
            if key not in gene_dict.keys():
                gene_dict[key] = [key, 0, 'Not in KB', 0, "NA"]
        unwanted = set(gene_dict.keys()) - set(genelist)
        for unwanted_key in unwanted:
            del gene_dict[unwanted_key]

    ### output the final prioritized associated gene list
    # Prioritize all found genes
    gene_dict = gene_prioritization(gene_dict)
    return gene_dict, std_output, weight_model