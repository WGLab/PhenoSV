import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='set up PhenoSV')
parser.add_argument('--path',type=str)


def change_path():
    global args
    args = parser.parse_args()
    df_path = os.path.join(args.path,"featuremaster_scu1026.csv")
    df = pd.read_csv(df_path)
    df_path_list = df['path'].tolist()
    df_path_list = [path.replace('/home/xu3/PhenoSV/data/', '') for path in df_path_list]
    #df_path_list = [os.path.join(args.path, path) for path in df_path_list]
    df['path'] = df_path_list
    target_path=os.path.join(args.path, "features_set.csv")
    print('saving feature file metadata to '+str(target_path))
    df.to_csv(target_path)
    #phenosv-light
    skip_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 20, 24, 28, 29, 31, 32, 33, 36,
                  38, 39, 40, 41, 45, 46, 47, 49, 52, 55, 56, 58]
    for i,p in enumerate(df_path_list):
        if i in skip_index:
            df_path_list[i] = 'skip'
    df['path'] = df_path_list
    target_path = os.path.join(args.path, "features_set_light.csv")
    print('saving feature file of PhenoSV-light metadata to ' + str(target_path))
    df.to_csv(target_path)
    feature_path = os.path.join(args.path, "features1026.csv")
    feature_df = pd.read_csv(feature_path)
    feature_df.bias = [0]*feature_df.shape[0]
    print('saving feature scaling file for PhenoSV-light metadata')
    feature_df.to_csv(os.path.join(args.path, "features1026_light.csv"))


if __name__ == '__main__':
    change_path()