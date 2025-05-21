import pandas as pd
import os
import glob
import json

from tqdm.auto import tqdm
from collections import defaultdict


def extract_subject_exclusive_annotations(
    data_dir: str,
    annotations_dir: str,
    n_subject_exclusive_folders: int, 
    verbose: bool = True,
):
    """
    AFAD annotations are organized in filenames.
    For each sample, the file name defines the attributes of the subject (age/gender/XXXXX-X.jpg).
    Age is encoded an integer. Gender could either be M (encoded as '/111/') and F (encoded as '/112/')
    """

    if verbose:
        print('Extracting Subject-Exclusive annotations from AFAD...')

    id2folder, current_folder = defaultdict(int), 0
    ages, genders, id_nums, img_paths, folders = [], [], [], [], []
    for sample_path in tqdm(glob.glob(data_dir + '/*/*/*.jpg'), disable=(not verbose), desc='Processing dataset image files'):
        attributes = sample_path.split(os.path.sep)

        age      = int(attributes[-3])
        gender   = 'M' if attributes[-2] == '111' else 'F'
        id_num   = int(attributes[-1].split('-')[0])
        img_path = os.path.relpath(sample_path, start=data_dir)

        if not id_num in id2folder:
            id2folder[id_num] = current_folder
            current_folder = (current_folder + 1) % n_subject_exclusive_folders
        
        folder = id2folder[id_num]

        img_paths.append(img_path)
        id_nums.append(id_num)
        ages.append(age)
        genders.append(gender)
        folders.append(folder)

    df = pd.DataFrame({
        'img_path': img_paths,
        'id_num': id_nums,
        'age': ages,
        'gender': genders,
        'folder': folders,
    })
    
    with open(annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))