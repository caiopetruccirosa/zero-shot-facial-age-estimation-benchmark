import pandas as pd
import os
import json

from scipy.io import loadmat  # this is the SciPy module that loads mat-files


def extract_subject_exclusive_annotations(
    data_dir: str,
    annotations_dir: str,
    verbose: bool = True,
):
    if verbose:
        print('Extracting Subject-Exclusive annotations from CACD2000...')

    mat_file        = loadmat(os.path.join(data_dir, 'celebrity2000_meta.mat'))
    image_data      = mat_file['celebrityImageData']
    meta_data       = mat_file['celebrityData']
    annotation_data = { attribute: image_data[attribute][0][0] for attribute in image_data.dtype.names }

    id2name = { id[0]: name[0][0] for name, id in zip(meta_data['name'][0][0], meta_data['identity'][0][0]) }

    img_paths = annotation_data['name'].flatten()
    id_nums   = annotation_data['identity'].flatten()
    ages      = annotation_data['age'].flatten()
    births    = annotation_data['birth'].flatten()
    lfws      = annotation_data['lfw'].flatten()
    ranks     = annotation_data['rank'].flatten()
    years     = annotation_data['year'].flatten()
    names     = [ id2name[id] for id in id_nums ]

    df = pd.DataFrame({
        'name': names,
        'img_path': img_paths,
        'id_num': id_nums,
        'age': ages,
        'birth': births,
        'lfw': lfws,
        'rank': ranks,
        'year': years,
    })

    df['folder'] = -1
    df.loc[df['rank']<=5, 'folder'] = 2
    df.loc[df['rank']<=2, 'folder'] = 1
    df.loc[df['rank']>5, 'folder']  = 0

    with open(annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))