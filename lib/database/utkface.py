import pandas as pd
import os
import glob
import json

from tqdm.auto import tqdm
from datetime import datetime


ANNOTATION_CORRECTIONS = {
    'part1/61_1_20170109142408075.jpg': [ 61, 'F', 'Black', str(datetime.strptime('20170109142408', '%Y%m%d%H%M%S')) ],
    'part1/61_3_20170109150557335.jpg': [ 61, 'F', 'Indian', str(datetime.strptime('20170109150557', '%Y%m%d%H%M%S')) ],
    'part2/39_1_20170116174525125.jpg': [ 39, 'M', 'Black', str(datetime.strptime('20170116174525', '%Y%m%d%H%M%S')) ],
    'part2/53__0_20170116184028385.jpg': [ 53, 'F', 'White', str(datetime.strptime('20170116184028', '%Y%m%d%H%M%S')) ],
}

RACEDICT = { 0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other' }


def extract_subject_exclusive_annotations(
    data_dir: str,
    annotations_dir: str,
    n_subject_exclusive_folders: int, 
    verbose: bool = True,
):
    """
    The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
    [age] is an integer from 0 to 116, indicating the age
    [gender] is either 0 (male) or 1 (female)
    [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

    Some files in UTKFace do not follow the [age]_[gender]_[race]_[date&time].jpg format
    They either omit some information and the leading '_' or omit the information only
    For this purpose, these files are manually corrected by the following dictionary
    """

    if verbose:
        print('Extracting Subject-Exclusive annotations from UTKFace...')

    all_files = [ f for file_type in ('/*.jpg', '/*.JPG', '/*.png', '/*.PNG') for f in glob.glob(data_dir + file_type) ]
    current_folder = 0
    img_paths, ages, genders, races, dateandtimes, folders = [], [], [], [], [], []
    for sample_path in tqdm(all_files, disable=(not verbose), desc='Processing dataset image files'):
        splitted_path = sample_path.split(os.path.sep)
        file_name     = os.path.join(splitted_path[-1], splitted_path[-2])
        basename, _   = os.path.splitext(splitted_path[-1])
        img_path      = os.path.relpath(sample_path, start=data_dir)

        if file_name in ANNOTATION_CORRECTIONS:
            age, gender, race, dateandtime = ANNOTATION_CORRECTIONS[file_name]
        else:
            attributes  = basename.split('_')
            age         = int(attributes[0])
            gender      = 'M' if int(attributes[1]) == 0 else 'F'
            race        = RACEDICT[int(attributes[2])]
            dateandtime = str(datetime.strptime(attributes[3][:14], '%Y%m%d%H%M%S'))
        

        img_paths.append(img_path)
        ages.append(age)
        genders.append(gender)
        races.append(race)
        dateandtimes.append(dateandtime)
        folders.append(current_folder)
        
        current_folder = (current_folder + 1) % n_subject_exclusive_folders

    df = pd.DataFrame({
        'img_path': img_path,
        'age': age,
        'race': race,
        'acquired': dateandtime,
        'folder': folders,
    })

    with open(annotations_dir + '/subject_exclusive_annotations.json', 'w+') as f:
        f.write(json.dumps(obj=df.to_dict(orient='records'), indent=4))
