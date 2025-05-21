import pandas as pd
import os
import glob
import json
import re

from tqdm.auto import tqdm


def extract_subject_exclusive_annotations(
    data_dir: str,
    annotations_dir: str,
    verbose: bool = True,
):
    """
    FGNET doesn't define a separate file with the annotations.
    For each sample, the file name defines the attributes of the subject (identity_age.jpg).
    E.g., 001a02.jpg (identity_age.jpg).
    Sometimes, the 'a' is lowercase, sometimes uppercase.
    If multiple images of the same person at the same age are given, they are distinguished by a trailing letter:
    E.g., 010A07a.jpg and 010A07b.jpeg.
    """

    if verbose:
        print('Extracting Subject-Exclusive annotations from FG-NET...')
    
    ages, numbers, img_paths = [], [], []
    all_files = [ f for ext in ('/*.jpg', '/*.JPG', '/*.png', '/*.PNG') for f in glob.glob(data_dir + ext) ]
    for sample_path in tqdm(all_files, disable=(not verbose), desc='Processing dataset image files'):
        attributes = re.findall(r'\d+', os.path.basename(sample_path))

        number   = int(attributes[0])
        age      = int(attributes[1])
        img_path = os.path.relpath(sample_path, start=data_dir)

        img_paths.append(img_path)
        numbers.append(number)
        ages.append(age)
    
    df = pd.DataFrame({
        'img_path': img_paths,
        'id_num': numbers,
        'age': ages,
        'folder': numbers, # assign identities to folders - each identity gets its own folder
    })

    with open(annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))
