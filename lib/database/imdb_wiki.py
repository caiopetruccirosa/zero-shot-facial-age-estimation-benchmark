import pandas as pd
import numpy as np
import os
import json
import re

from tqdm.auto import tqdm
from collections import defaultdict


IMDB_WIKI_BASE_DIR         = 'IMDB_WIKI/imdb/'
IMDB_WIKI_ANNOTATIONS_DIR  = 'benchmark/databases/IMDB-WIKI'
IMDB_CLEAN_BASE_DIR        = '.....'
IMDB_CLEAN_ANNOTATIONS_DIR = 'benchmark/databases/IMDB-Clean'


def extract_imdb_wiki_fp_age_subject_exclusive_annotations(data_root_dir: str, verbose: bool = True):
    if verbose:
        print('Extracting Subject-Exclusive annotations from IMDB-WIKI...')

    imdb_clean_dir = os.path.join(data_root_dir, IMDB_CLEAN_BASE_DIR)

    attribute_names = ['img_path', 'age', 'gender', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'roll', 'yaw', 'pitch']

    df_train = pd.read_csv(os.path.join(imdb_clean_dir, 'imdb_train_new.csv'), sep=',', names=attribute_names, skiprows=[0])
    df_val   = pd.read_csv(os.path.join(imdb_clean_dir, 'imdb_valid_new.csv'), sep=',', names=attribute_names, skiprows=[0])
    df_test  = pd.read_csv(os.path.join(imdb_clean_dir, 'imdb_test_new.csv'), sep=',', names=attribute_names, skiprows=[0])

    df_train['folder'] = 0
    df_val['folder']   = 1
    df_test['folder']  = 2

    df = pd.concat([df_train, df_val, df_test], ignore_index=False, axis=0)

    get_name = lambda row: re.search(r'[0-9]+\/nm([0-9]+)_rm', row['img_path'])[1] # type: ignore
    df['name'] = df.apply(get_name, axis=1)

    df.img_path = IMDB_WIKI_BASE_DIR + df.img_path

    df['bbox'] = [
        [int(__) for __ in _] 
        for _ in np.stack([
            df.bbox_left,
            df.bbox_top,
            df.bbox_right,
            df.bbox_top,
            df.bbox_right,
            df.bbox_bottom,
            df.bbox_left,
            df.bbox_bottom,
        ]).T.astype(int)]
    df.drop(['bbox_left', 'bbox_right', 'bbox_bottom', 'bbox_top'], inplace=True, axis=1)

    with open(IMDB_CLEAN_ANNOTATIONS_DIR + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))


def extract_imdb_wiki_em_cnn_subject_exclusive_annotations(data_root_dir: str, n_subject_exclusive_folder: int, verbose: bool = True):
    if verbose:
        print('Extracting Subject-Exclusive annotations from IMDB-WIKI...')

    imdb_wiki_dir = os.path.join(data_root_dir, IMDB_WIKI_BASE_DIR)

    attribute_names = ['img_path', 'name', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'age', 'gender', 'confidence']

    df = pd.read_csv(os.path.join(imdb_wiki_dir, 'emcnn_annotion_18-Nov-2017.txt'), sep='\t', names=attribute_names)
    
    # keep only files with confidence over the threshold
    confidence_threshold = 0.0
    df = df.loc[df['confidence'] >= confidence_threshold]

    # prepend dataset path to file paths
    df.img_path = IMDB_WIKI_BASE_DIR + os.sep + df.img_path

    print('Concatenating bounding box annotations ...')
    # Build bounding box annotation
    df['bbox'] = [
        [ int(coord) for coord in bbox ] 
        for bbox in np.stack([
            df.bbox_left,
            df.bbox_top,
            df.bbox_right,
            df.bbox_top,
            df.bbox_right,
            df.bbox_bottom,
            df.bbox_left,
            df.bbox_bottom,
        ]).T.astype(int)
    ]
    df.drop(['bbox_left', 'bbox_right', 'bbox_bottom', 'bbox_top'], inplace=True, axis=1)

    # Prepare list of unique identities
    unique_ids = df.name.unique()
    np.random.shuffle(unique_ids)

    # Assign identities to folders

    id2folder, folders, current_folder = defaultdict(int), [], 0
    for name in tqdm(df.name, disable=(not verbose), desc='Assigning subjects to folders'):
        if name not in id2folder:
            id2folder[name] = current_folder
            current_folder = (current_folder + 1) % n_subject_exclusive_folder
        folders.append(id2folder[name])

    df['folder'] = folders

    with open(IMDB_WIKI_ANNOTATIONS_DIR + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))