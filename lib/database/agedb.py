import pandas as pd
import os
import glob
import json

from tqdm.auto import tqdm
from collections import defaultdict
from lib.database.utils import assign_folders_with_similar_age_distribution


def extract_subject_exclusive_annotations(
    agedb_base_dir: str,
    agedb_annotations_dir: str,
    data_root_dir: str, 
    n_subject_exclusive_folder: int, 
    verbose: bool = True,
):
    """
    AgeDB annotations are organized in filenames.
    For each sample, the file name defines the attributes of the subject (number_name_age_gender.jpg).
    Age is encoded an integer. Gender could either be 'm' and 'f'.
    E.g., 0_MariaCallas_35_f.jpg (number_name_age_gender.jpg)
    """

    if verbose:
        print('Extracting Subject-Exclusive annotations from AgeDB...')
        
    agedb_dir = os.path.join(data_root_dir, agedb_base_dir)

    id2ages, names, numbers, ages, genders, img_paths = defaultdict(list), [], [], [], [], []
    for sample_path in tqdm(glob.glob(agedb_dir + '/*.jpg'), disable=(not verbose), desc='Processing dataset image files'):
        attributes = os.path.basename(sample_path).split('_')

        number   = int(attributes[0])
        name     = attributes[1]
        age      = int(attributes[2])
        gender   = 'M' if 'm' in attributes[3] else 'F'
        img_path = os.path.relpath(sample_path, start=data_dir)

        img_paths.append(img_path)
        names.append(name)
        ages.append(age)
        genders.append(gender)
        numbers.append(number)
        id2ages[name].append(age)
    
    folders = assign_folders_with_similar_age_distribution(names, id2ages, n_subject_exclusive_folder, verbose)

    df = pd.DataFrame({
        'img_path': img_paths,
        'name': names,
        'age': ages,
        'gender': genders,
        'number': numbers,
        'folder': folders,
    })

    with open(agedb_annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))