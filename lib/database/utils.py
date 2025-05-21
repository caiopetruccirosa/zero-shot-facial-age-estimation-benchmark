import pandas as pd
import numpy as np

from collections import defaultdict, Counter
from tqdm.auto import tqdm


def get_supported_databases_data_directory() -> dict[str, str]: 
    return {
        'AFAD': 'AFAD/AFAD-Full',
        'AgeDB': 'AgeDB/AgeDB',
        'CACD2000': 'CACD2000/CACD2000',
        'CLAP2016': 'appa-real-release',
        'FG-NET': 'FG-NET/images',
        'IMDB-WIKI': '',
        'MORPH': 'MORPH',
        'UTKFace': 'UTKFace',
    }


def assert_subject_exclusivity(df: pd.DataFrame, id_key: str, folder_key: str):
    assert (df.groupby(id_key)[folder_key].nunique() == 1).all(), 'Folders assigned are not subject exclusive!'


def assign_folders_with_similar_age_distribution(samples_id: list[int], id2ages: dict[int, list[int]], n_subject_exclusive_folder: int, verbose: bool) -> list[int]:
    # splits the files into subject exclusive folders with balanced distributions of age
    id_n_ages   = list(id2ages.items())
    id2folder   = defaultdict(int)
    folder2ages = defaultdict(lambda: np.array([]))

    np.random.shuffle(id_n_ages)

    for subject_id, subject_ages in tqdm(id_n_ages, total=len(id_n_ages), disable=(not verbose), desc='Assigning subjects to folders'):
        (most_represented_age, _) = Counter(subject_ages).most_common(n=1)

        # for each folder, count how many examples of that age are already there
        age_count_per_folder = [ sum(folder2ages[folder] == most_represented_age) for folder in range(n_subject_exclusive_folder) ]

        # pick the folder with the smallest count of that age
        folder_with_least_counts = np.argmin(age_count_per_folder)

        # assign this subject to that folder
        id2folder[subject_id] = int(folder_with_least_counts)

        # append all of this subject's ages into that folder's age list
        folder2ages[folder_with_least_counts] = np.append(folder2ages[folder_with_least_counts], subject_ages)
    
    folders = [ id2folder[name] for name in samples_id ]
    
    return folders