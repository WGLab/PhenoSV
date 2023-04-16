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
    df_path_list = [os.path.join(args.path, path) for path in df_path_list]
    df['path'] = df_path_list
    target_path=os.path.join(args.path, "features_set.csv")
    print('saving feature file metadata to '+str(target_path))
    df.to_csv(target_path)


if __name__ == '__main__':
    change_path()