import argparse
import os

from lib.database import afad, agedb, cacd2000, clap2016, fgnet, morph, utkface
from lib.database.utils import get_supported_databases_data_directory


def main():
    database_data_dirs  = get_supported_databases_data_directory()
    supported_databases = database_data_dirs.keys()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--database', help='', type=str, required=True, choices=supported_databases)
    arg_parser.add_argument('--data_root_dir', help='', type=str, required=True)
    arg_parser.add_argument('--n_subject_exclusive_folders', help='', type=int, required=False)
    arg_parser.add_argument('--quiet', '-q', help='', action='store_true', required=False)
    args = arg_parser.parse_args()

    if args.database in ['AFAD', 'AgeDB', 'MORPH', 'UTKFace']:
        assert args.n_subject_exclusive_folders is not None, f'To extract annotations for {args.database} database, --n_subject_exclusive_folders flag is required.'

    annotations_dir = f'benchmark/databases/{args.database}'
    data_dir        = os.path.join(args.data_root_dir, database_data_dirs[args.database])
    match args.database:
        case 'AFAD':
            afad.extract_subject_exclusive_annotations(data_dir, annotations_dir, args.n_subject_exclusive_folders, not args.quiet)
        case 'AgeDB':
            agedb.extract_subject_exclusive_annotations(data_dir, annotations_dir, args.n_subject_exclusive_folders, not args.quiet)
        case 'CACD2000':
            cacd2000.extract_subject_exclusive_annotations(data_dir, annotations_dir, not args.quiet)
        case 'CLAP2016':
            clap2016.extract_subject_exclusive_annotations(data_dir, annotations_dir, not args.quiet)
        case 'FG-NET':
            fgnet.extract_subject_exclusive_annotations(data_dir, annotations_dir, not args.quiet)
        case 'IMDB-WIKI':
            pass
        case 'MORPH':
            morph.extract_subject_exclusive_annotations(data_dir, annotations_dir, args.n_subject_exclusive_folders, not args.quiet)
        case 'UTKFace':
            utkface.extract_subject_exclusive_annotations(data_dir, annotations_dir, args.n_subject_exclusive_folders, not args.quiet)
    

if __name__ == '__main__':
    main()