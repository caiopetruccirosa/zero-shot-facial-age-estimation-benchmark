import pandas as pd
import os
import json


def extract_subject_exclusive_annotations(
    data_dir: str,
    annotations_dir: str,
    verbose: bool = True,
):
    if verbose:
        print('Extracting Subject-Exclusive annotations from CLAP2016...')
        
    df_train = pd.read_csv(os.path.join(data_dir, 'gt_avg_train.csv'))
    df_val   = pd.read_csv(os.path.join(data_dir, 'gt_avg_valid.csv'))
    df_test  = pd.read_csv(os.path.join(data_dir, 'gt_avg_test.csv'))

    df_train['folder'] = 0
    df_val['folder']   = 1
    df_test['folder']  = 2

    df_train['file_name'] = [ f'train/{file_name}' for file_name in df_train['file_name'] ]
    df_val['file_name']   = [ f'valid/{file_name}' for file_name in df_val['file_name'] ]
    df_test['file_name']  = [ f'test/{file_name}' for file_name in df_test['file_name'] ]

    df = pd.concat([ df_train, df_val, df_test ], ignore_index=False, axis=0)
    df = df.rename(columns={'file_name': 'img_path', 'real_age': 'age'})
    df = df.drop(['num_ratings', 'apparent_age_avg', 'apparent_age_std'], axis=1)

    with open(annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))