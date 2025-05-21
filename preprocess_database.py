import argparse
import os
import json
import cv2
    
from tqdm.auto import tqdm

from lib.face_detection.retinaface import (
    get_retina_face_model,
    get_face_from_landmarks,
)
from lib.database.utils import get_supported_databases_data_directory
from lib.utils import padded_number

def main():
    database_data_dirs = get_supported_databases_data_directory()
    supported_databases = database_data_dirs.keys()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--database', help='', type=str, required=True, choices=supported_databases)
    arg_parser.add_argument('--data_root_dir', help='', type=str, required=True)
    arg_parser.add_argument('--image_size', help='', type=int, required=True, nargs='*')
    arg_parser.add_argument('--quiet', '-q', help='', action='store_true', required=False)
    args = arg_parser.parse_args()

    w, h                 = tuple(args.image_size)
    eye2eye_scale        = 1.92
    eye2mouth_scale      = 1.89
    detection_threshold  = 0.5
    detection_size       = (w, h)

    # instantiate retina face model for detection
    retina_face_model = get_retina_face_model(detection_size, detection_threshold)

    # create directories from annotation and data
    annotations_dir          = f'benchmark/databases/{args.database}'
    data_dir                 = os.path.join(args.data_root_dir, database_data_dirs[args.database])
    processed_annotations_dir = f'{annotations_dir}/data/subject_exclusive_{w}x{h}_retinaface'
    processed_data_dir        = f'{processed_annotations_dir}/images'

    os.makedirs(processed_annotations_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # load dataset annotations
    with open(annotations_dir + '/subject_exclusive_annotations.json', 'r') as f:
        annotations = json.load(f)

    # preprocess dataset and store new annotations
    processed_annotations = []
    for i in tqdm(range(len(annotations)), disable=args.quiet, desc=f'Detecting and cropping faces of {args.database} database'):
        sample = annotations[i]
        img_path, age, folder = sample['img_path'], sample['age'], sample['folder']
        
        # check if sample's image path exists
        abs_img_path = f'{data_dir}/{img_path}'
        if not os.path.exists(abs_img_path):
            tqdm.write(f'[Sample {i}] Image with path does not exists. Image path: {img_path}')
            continue

        # check if sample's image exists
        img = cv2.imread(abs_img_path)
        if img is None:
            tqdm.write(f'[Sample {i}] Empty or corrupted image found. Image path: {img_path}')
            continue

        face_img, aligned_bbox, success = get_face_from_landmarks(
            retina_face_model=retina_face_model, 
            full_img=img, 
            face_img_size=(w, h),
            eye2mouth_scale=eye2mouth_scale, 
            eye2eye_scale=eye2eye_scale, 
            adaptive_size=True,
        )
        if not success:
            tqdm.write(f'[Sample {i}] No detected face using landmarks. Image path: {img_path}')
            continue
            
        face_img_file_name = f'img{padded_number(i, len(annotations))}.jpg'
        face_img_path      = f'{processed_data_dir}/{face_img_file_name}'
        cv2.imwrite(face_img_path, face_img)

        # create new entry for processed sample
        new_sample = {
            'id': i,
            'folder': folder,
            'age': age,
            'full_img_path': img_path,
            'face_img_path': face_img_file_name,
            'face_aligned_bbox': aligned_bbox
        }
        processed_annotations.append(new_sample)

    print(f'Accepted {len(processed_annotations)} faces out of {len(annotations)}')

    with open(processed_annotations_dir + '/subject_exclusive_annotations.json', 'w') as f:
        json.dump(processed_annotations, f)


if __name__ == '__main__':
    main()