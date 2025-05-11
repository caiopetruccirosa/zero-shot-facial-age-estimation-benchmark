import argparse
import os
import json
import cv2
    
from argparse import Namespace
from rich.progress import track
from preprocessing.face_detection import (
    get_retina_face_model,
    get_face_from_landmarks,
)


def get_arguments() -> Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--database', help='', type=str, required=True)
    arg_parser.add_argument('--image_size', help='', type=int, required=True, nargs='*')
    arg_parser.add_argument('--data_dir', help='', type=str, required=True)
    return arg_parser.parse_args()


def main():
    # read arguments
    args = get_arguments()
    image_size = tuple(args.image_size)
    database = args.database

    # load database raw annotations
    annotations_file = f'databases/{database}/annotations.json'
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # create directories for storing preprocessed dataset
    w, h = image_size
    new_data_dir = f'databases/{database}/data/{w}x{h}_RetinaFace_Faces'

    os.makedirs(new_data_dir, exist_ok=True)
    os.makedirs(f'{new_data_dir}/images', exist_ok=True)

    # create auxiliary print error message
    print_error_message = lambda sample_idx, img_path, message: print(f'[Sample {sample_idx}] {message}. Image path: {img_path}')

    eye2eye_scale                = 1.92
    eye2mouth_scale              = 1.89
    detection_size               = image_size
    detection_threshold          = 0.5

    # instantiate retina face model for detection
    retina_face_model = get_retina_face_model(detection_size, detection_threshold)

    # preprocess dataset and store new annotations
    new_annotations = []
    for i, sample in track(enumerate(annotations), total=len(annotations)):
        # check if sample's image path exists
        img_path = f'{args.data_dir}/{sample['img_path']}'
        if not os.path.exists(img_path):
            print_error_message(i, sample['img_path'], 'Image with path does not exists')
            continue

        # check if sample's image exists
        img = cv2.imread(img_path)
        if img is None:
            print_error_message(i, sample['img_path'], 'Empty or corrupted image found')
            continue

        face_img, aligned_bbox, success = get_face_from_landmarks(
            retina_face_model=retina_face_model, 
            full_img=img, 
            face_img_size=image_size,
            eye2mouth_scale=eye2mouth_scale, 
            eye2eye_scale=eye2eye_scale, 
            adaptive_size=True,
        )
        if not success:
            print_error_message(i, sample['img_path'], 'No detected face using landmarks')
            continue
            
        face_img_relative_path = f'images/img{i:07d}.jpg'
        face_img_absolute_path = f'{new_data_dir}/{face_img_relative_path}'
        cv2.imwrite(face_img_absolute_path, face_img)

        # create new entry for processed sample
        new_sample = {
            'id': i,
            'folder': sample['folder'],
            'age': sample['age'],
            'full_img_path': sample['img_path'],
            'face_img_path': face_img_relative_path,
            'face_aligned_bbox': aligned_bbox
        }
        new_annotations.append(new_sample)

    print(f'Accepted {len(new_annotations)} faces out of {len(annotations)}')

    new_annotations_file = f'{new_data_dir}/annotations.json'
    with open(new_annotations_file, 'w') as f:
        json.dump(new_annotations, f)


if __name__ == '__main__':
    main()