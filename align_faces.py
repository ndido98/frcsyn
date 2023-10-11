"""
This script is responsible for aligning the faces in each image found in the specified directory.
The user can specify the margin and image size for the aligned faces,
and they will be saved in a new directory, following the same structure as the original.
The chosen bounding box is the largest one closest to the center of the image.
"""

import argparse
import os
import logging
from pathlib import Path
import torch.multiprocessing as mp

from tqdm import tqdm
import numpy as np
import cv2 as cv
import torch
from facenet_pytorch import MTCNN


ALLOWED_EXTENSIONS = ["jpg", "png"]


def multi_rglob(path: Path, extensions: list[str]) -> list[Path]:
    """
    Recursively find all files in the specified directory with the specified extensions.
    """
    files = []
    # Recursively walk the file system
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            # Check if the file has the specified extension
            if any([filename.endswith(ext) for ext in extensions]):
                files.append(Path(root) / filename)
    return files


def multi_glob(path: Path, extensions: list[str]) -> list[Path]:
    """
    Find all files in the specified directory with the specified extensions.
    """
    files = []
    # Walk the file system
    for filename in os.listdir(path):
        # Check if the file has the specified extension
        if any([filename.endswith(ext) for ext in extensions]):
            files.append(path / filename)
    return files


def select_best_bbox(img_shape: np.ndarray, bboxes: np.ndarray) -> tuple[int, np.ndarray]:
    # bboxes is a (N, 4) array, where N is the number of detected faces
    # and each value is the bounding box coordinates
    if bboxes.ndim == 1:
        bboxes = bboxes[None, :]
    # bboxes_areas is a (N,) array, where N is the number of detected faces
    # and each value is the area of the bounding box
    bboxes_areas = np.prod(bboxes[..., 2:] - bboxes[..., :2], axis=1)
    # bboxes_centers is a (N, 2) array, where N is the number of detected faces
    # and each value is the center of the bounding box
    bboxes_centers = (bboxes[..., 2:] + bboxes[..., :2]) / 2
    # image_center is a (2,) array, where each value is the center of the image
    image_center = img_shape / 2
    # bboxes_distances is a (N,) array, where N is the number of detected faces
    # and each value is the Euclidean distance between the center of the image
    # and the center of the bounding box
    bboxes_distances = np.linalg.norm(bboxes_centers - image_center[None, :], axis=1)
    # bboxes_scores is a (N,) array, where N is the number of detected faces
    # and each value is the score of the bounding box
    bboxes_scores = bboxes_areas - bboxes_distances ** 2
    # Choose the bounding box with the highest score
    best_bbox_index = np.argmax(bboxes_scores)
    return best_bbox_index, bboxes[best_bbox_index]


def obb_to_aabb(obb: np.ndarray) -> np.ndarray:
    bottom, top = np.min(obb[:, 1]), np.max(obb[:, 1])
    left, right = np.min(obb[:, 0]), np.max(obb[:, 0])
    return np.array([left, bottom, right, top])


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv.transform(points.reshape(1, -1, 2), matrix).reshape(-1, 2).squeeze()


def transform_bbox(obb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Transform the bounding box into a polygon
    x1, y1, x2, y2 = obb
    obb = np.array([
        [x1, y1],
        [x1, y2],
        [x2, y2],
        [x2, y1],
    ])
    # Transform the polygon
    obb = transform_points(obb, matrix)
    # Convert the polygon back to a bounding box
    aabb = obb_to_aabb(obb)
    return obb, aabb


def get_img_bbox_intersection(img_shape: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    # bbox is a (4,) array, where each value is the bounding box coordinates
    # img_shape is a (2,) array, where each value is the image shape
    return np.array([
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(img_shape[1], bbox[2]),
        min(img_shape[0], bbox[3]),
    ])


def align_face(args: tuple[Path, Path, Path, int, int, bool]) -> bool:
    input_file, input_root, output_root, margin, size, align = args
    output_file = output_root / input_file.relative_to(input_root)
    if output_file.exists():
        return True
    img = cv.imread(str(input_file))
    if img is None:
        logging.warning(f"Could not read {input_file}, skipping")
        return False
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_shape = np.array(img.shape[:2]).astype(np.float32)
    detector = MTCNN(select_largest=False, device="cpu")
    detect_failure = False
    try:
        img_boxes, _, landmarks = detector.detect(img, landmarks=True)
        if img_boxes is None:
            detect_failure = True
    except RuntimeError:
        detect_failure = True
    if detect_failure:
        logging.warning(f"Could not detect any face in {input_file}, skipping")
        return False
    img_boxes = np.asarray(img_boxes, dtype=np.float32)
    img_bbox_idx, img_bbox = select_best_bbox(img_shape, img_boxes)
    img_landmarks = landmarks[img_bbox_idx].astype(np.float32)
    if align:
        left_eye, right_eye = img_landmarks[0], img_landmarks[1]
        eye_diff = right_eye - left_eye
        rotation_degrees = np.degrees(np.arctan2(eye_diff[1], eye_diff[0]))
        rotation_center = np.array([img_shape[1] / 2, img_shape[0] / 2], dtype=np.float32)
        # Rotate the image so that the eyes are aligned horizontally
        rotation_matrix = cv.getRotationMatrix2D(tuple(rotation_center), rotation_degrees, 1.0)
        # Compute the new image size
        rotation_matrix_cos, rotation_matrix_sin = np.abs(rotation_matrix[0, 0]), np.abs(rotation_matrix[0, 1])
        img_height, img_width = img_shape[:2]
        new_img_height, new_img_width = (
            int(img_height * rotation_matrix_cos + img_width * rotation_matrix_sin),
            int(img_height * rotation_matrix_sin + img_width * rotation_matrix_cos),
        )
        # Adjust the rotation matrix to take into account the translation
        rotation_matrix[0, 2] += new_img_width / 2 - rotation_center[0]
        rotation_matrix[1, 2] += new_img_height / 2 - rotation_center[1]
        img = cv.warpAffine(img, rotation_matrix, (new_img_width, new_img_height))
        img_shape = np.array(img.shape[:2]).astype(np.float32)
        # Transform the landmarks and bounding box
        _, img_bbox = transform_bbox(img_bbox, rotation_matrix)
    # Extend the bounding boxes by the specified margin in all directions
    img_bbox[:2] = img_bbox[:2] - margin / 2
    img_bbox[2:] = img_bbox[2:] + margin - margin / 2
    # Make the bounding box square maintaining the center
    img_bbox_width, img_bbox_height = img_bbox[2:] - img_bbox[:2]
    img_bbox_size = max(img_bbox_width, img_bbox_height)
    if img_bbox_width > img_bbox_height:
        img_bbox[1] -= (img_bbox_size - img_bbox_height) / 2
        img_bbox[3] += (img_bbox_size - img_bbox_height) / 2
    else:
        img_bbox[0] -= (img_bbox_size - img_bbox_width) / 2
        img_bbox[2] += (img_bbox_size - img_bbox_width) / 2
    x1, y1, x2, y2 = np.round(img_bbox).astype(np.int32)
    # Pad the image with black borders so that the bounding box is not out of bounds
    cropped = np.zeros((y2 - y1, x2 - x1, 3), dtype=img.dtype)
    i_x1, i_y1, i_x2, i_y2 = np.round(get_img_bbox_intersection(img_shape, img_bbox)).astype(np.int32)
    img_roi = img[i_y1:i_y2, i_x1:i_x2]
    cropped[i_y1 - y1:i_y2 - y1, i_x1 - x1:i_x2 - x1] = img_roi
    if cropped is None or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        logging.warning(f"Unable to crop {input_file} (bounding box: {img_bbox}, cropped shape: {cropped.shape}), skipping")
        return False
    interpolation = cv.INTER_CUBIC if size * size > cropped.shape[0] * cropped.shape[1] else cv.INTER_AREA
    # Resize the cropped image to the specified size
    resized = cv.resize(cropped, (size, size), interpolation=interpolation)
    # Save the resized image in the output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    resized = cv.cvtColor(resized, cv.COLOR_RGB2BGR)
    cv.imwrite(str(output_file), resized)
    return True


def align_faces(input_dir: Path, output_dir: Path, recursive: bool, margin: int, size: int, align: bool, num_workers: int) -> None:
    logging.info(f"Starting face alignment in {input_dir}")
    if recursive:
        files = multi_rglob(input_dir, ALLOWED_EXTENSIONS)
    else:
        files = multi_glob(input_dir, ALLOWED_EXTENSIONS)
    logging.info(f"Found {len(files)} images")
    logging.info("Aligning faces...")
    total_success = 0
    elems = [(file, input_dir, output_dir, margin, size, align) for file in files]
    if num_workers == 0:
        for elem in tqdm(elems):
            success = align_face(elem)
            total_success += 1 if success else 0
    else:
        with mp.Pool(num_workers) as pool:
            for success in tqdm(pool.imap_unordered(align_face, elems), total=len(elems)):
                total_success += 1 if success else 0
    logging.info(f"Done! Aligned {total_success} images, skipped {len(files) - total_success} images (success rate: {total_success / len(files) * 100:.2f}%)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Align faces in images")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Directory containing the images to be aligned",
        dest="input_dir",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory where the aligned images will be saved",
        dest="output_dir",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=False,
        help="Whether to recursively search for images in the input directory or not",
    )
    parser.add_argument(
        "-m", "--margin",
        type=int,
        default=0,
        help="Margin to be added to the bounding box",
    )
    parser.add_argument(
        "-s", "--image-size",
        type=int,
        default=160,
        help="Size of the aligned images",
        dest="image_size",
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=0,
        help="Number of workers to be used for the alignment",
        dest="num_workers",
    )
    parser.add_argument(
        "-a", "--align",
        action="store_true",
        default=False,
        help="Whether to align the faces or not using the eyes line",
    )
    args = parser.parse_args()
    align_faces(
        Path(args.input_dir) if isinstance(args.input_dir, str) else args.input_dir,
        Path(args.output_dir) if isinstance(args.output_dir, str) else args.output_dir,
        args.recursive,
        args.margin,
        args.image_size,
        args.align,
        args.num_workers,
    )