import numpy as np
import warnings
import cv2

from insightface.app import FaceAnalysis
from insightface.model_zoo import RetinaFace


def get_retina_face_model(detection_size: tuple[int, int], detection_threshold: float) -> RetinaFace:
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=detection_size, det_thresh=detection_threshold)
    return app.models['detection']


def get_face_from_landmarks(
    retina_face_model: RetinaFace, 
    full_img: np.ndarray, 
    face_img_size: tuple[int, int],
    eye2eye_scale: float,
    eye2mouth_scale: float,
    adaptive_size: bool = True, 
) -> tuple[np.ndarray, list[int], bool]:
    faces_landmarks = get_faces_landmarks(retina_face_model, full_img, adaptive_size)
    if len(faces_landmarks) == 0:
        return np.empty(shape=0), [], False

    faces_aligned_bboxes = [ 
        get_aligned_bounding_box(lm['mouth_avg'], lm['eye_left'], lm['eye_right'], eye2eye_scale, eye2mouth_scale).tolist() 
        for lm in faces_landmarks 
    ]
    faces_areas = [ 
        calculate_polygon_area(bbox) 
        for bbox in faces_aligned_bboxes
    ]

    largest_face_idx = np.argmax(faces_areas)
    face_aligned_bbox = faces_aligned_bboxes[largest_face_idx]
    face_img = crop_image(full_img, face_aligned_bbox, face_img_size)

    return face_img, face_aligned_bbox, True


def get_faces_landmarks(retina_face_model: RetinaFace, img: np.ndarray, adaptive_size: bool):
    # detection with original config size
    _, all_default_landmarks = retina_face_model.detect(img)
    all_default_landmarks = all_default_landmarks.astype(int) # type: ignore

    if not adaptive_size:
        return all_default_landmarks
    
    # detection with adaptive size
    adapted_size = (np.maximum(np.array(img.shape[:2])//32, 1) * 32).astype(int)
    original_size = retina_face_model.input_size
    retina_face_model.input_size = adapted_size

    _, all_adaptive_landmarks = retina_face_model.detect(img)
    all_adaptive_landmarks = all_adaptive_landmarks.astype(int) # type: ignore

    retina_face_model.input_size = original_size
    all_landmarks = np.concatenate([all_adaptive_landmarks, all_default_landmarks], axis=0)

    all_target_landmarks = [ 
        { 'eye_right': lm[0], 'eye_left': lm[1], 'mouth_avg': (lm[3]+lm[4])/2, } 
        for lm in all_landmarks 
    ]

    return all_target_landmarks


def calculate_polygon_area(points):
    """calculate area using shoelace theorem."""
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    return 0.5 * abs(sum((x1*y2)-(x2*y1) for (x1, y1), (x2, y2) in zip(points, np.roll(points, shift=-1, axis=0))))


def get_aligned_bounding_box(
    mouth_avg: np.ndarray, 
    eye_left: np.ndarray, 
    eye_right: np.ndarray,
    eye2eye_scale: float,
    eye2mouth_scale: float,
) -> np.ndarray:
    """Calculate alignment transformation based on facial landmarks."""
    eye_center_vec = (eye_left + eye_right) * 0.5
    eye_to_eye_vec = eye_left - eye_right
    eye_to_mouth_vec = mouth_avg - eye_center_vec

    # Validate eye direction
    rotated_eye_to_mouth_vec = np.array([eye_to_mouth_vec[1], -eye_to_mouth_vec[0]])
    if np.dot(rotated_eye_to_mouth_vec, eye_to_eye_vec) < 0:
        warnings.warn("Adjusting eye direction based on mouth position")
        eye_to_eye_vec = -eye_to_eye_vec

    # calculate transformation basis
    x = eye_to_eye_vec - (np.flipud(eye_to_mouth_vec) * [-1, 1])
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye_vec)*eye2eye_scale, np.hypot(*eye_to_mouth_vec)*eye2mouth_scale)
    y = np.flipud(x) * [-1, 1]

    center = eye_center_vec + eye_to_mouth_vec * 0.1
    
    return np.stack([center-x-y, center+x-y, center+x+y, center-x+y]).flatten().astype(int)


def crop_image(
    img: np.ndarray,
    bbox: list[int],
    out_size: tuple[int, int],
    margin: tuple[float, float] = (0, 0),
    one_based_bbox: bool = True
) -> np.ndarray:
    # Reshape bbox to 4x2 matrix and convert to float32
    quad = np.array(bbox, dtype=np.float32).reshape(4, 2) - (1 * one_based_bbox)

    # Calculate margin expansion vectors
    A, B, C, D = quad
    ext_A = A + ((A-B) * margin[0]) + ((A-D) * margin[1])
    ext_B = B + ((B-A) * margin[0]) + ((B-C) * margin[1])
    ext_C = C + ((C-D) * margin[0]) + ((C-B) * margin[1])

    # Select three non-colinear points for affine transform
    src_points = np.array([ext_A, ext_B, ext_C])
    dst_points = np.array([[0, 0], [out_size[0]-1, 0], [out_size[0]-1, out_size[1]-1]], dtype=np.float32)

    # Compute transformation
    M = cv2.getAffineTransform(src_points, dst_points)
    cropped_img = cv2.warpAffine(img, M, out_size)

    return cropped_img