import pandas as pd
import os
import json

from lib.database.utils import assign_folders_with_similar_age_distribution


def extract_subject_exclusive_annotations(
    data_dir: str,
    annotations_dir: str,
    n_subject_exclusive_folders: int, 
    verbose: bool = True,
):
    if verbose:
        print('Extracting Subject-Exclusive annotations from MORPH...')

    df = pd.read_csv(os.path.join(data_dir, 'morph_2008_nonCommercial.csv'))

    ids     = df.id_num.to_list()
    id2ages = { id_num: df.age[df.id_num == id_num].to_list() for id_num in df.id_num.unique() }
    folders = assign_folders_with_similar_age_distribution(ids, id2ages, n_subject_exclusive_folders, verbose)

    df = df.rename(columns={'photo': 'img_path'})
    df['img_path'] = df['img_path'].apply(lambda img_path: os.path.relpath(img_path, start=data_dir))
    df['folder'] = folders

    with open(annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))