# Import typing modules for type annotations
from typing import List, Tuple

# Import OpenCV for image processing, NumPy for numerical operations
import cv2
import numpy as np

# Import model classes for barcode detection and OCR (assumed to be custom modules)
from detection.detect import Yolov9
from ocr.model import OcrModel


def rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate an image by a given angle using OpenCV.

    Args:
        image (np.ndarray): The input image to rotate.
        angle (float): The rotation angle in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the rotated image and the rotation matrix.
    """
    # Get the dimensions of the image (height and width)
    (h, w) = image.shape[:2]
    # Compute the center point of the image
    center = (w / 2, h / 2)
    # Compute the rotation matrix for the given angle with no scaling (scale factor = 1.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Calculate the cosine and sine of the rotation angle (in absolute value)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # Compute the new width and height of the rotated image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # Adjust the rotation matrix to account for the translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    # Perform the actual rotation and get the rotated image
    rotated = cv2.warpAffine(image, M, (nW, nH))
    # Return the rotated image and the rotation matrix
    return rotated, M


def invert_transformation_matrix(M: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of an affine transformation matrix.

    Args:
        M (np.ndarray): The affine transformation matrix to invert.

    Returns:
        np.ndarray: The inverse of the affine transformation matrix.
    """
    # Use OpenCV function to invert the affine transformation matrix
    Minv = cv2.invertAffineTransform(M)
    # Return the inverse matrix
    return Minv


def transform_bounding_boxes(
        bboxes: List[List[float]],
        Minv: np.ndarray,
        original_shape: Tuple[int, int, int]
) -> List[List[float]]:
    """
    Transform bounding boxes back to the original image coordinates.

    Args:
        bboxes (List[List[float]]): List of bounding boxes from the rotated image.
        Minv (np.ndarray): The inverse transformation matrix.
        original_shape (Tuple[int, int, int]): The shape of the original image.

    Returns:
        List[List[float]]: List of transformed bounding boxes in original image coordinates.
    """
    # Initialize a list to store the transformed bounding boxes
    transformed_bboxes = []
    # Extract the height and width from the original image shape
    height, width = original_shape[:2]
    # Iterate over each bounding box in the list
    for bbox in bboxes:
        # Unpack the bounding box coordinates and confidence score
        x1, y1, x2, y2, conf = bbox
        # Create an array of the four corner points of the bounding box
        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='float32')
        # Apply the inverse transformation to the points
        transformed_points = cv2.transform(np.array([points]), Minv)[0]
        # Get the minimum and maximum x and y coordinates from the transformed points
        new_x1, new_y1 = np.min(transformed_points, axis=0)
        new_x2, new_y2 = np.max(transformed_points, axis=0)
        # Ensure the coordinates are within the boundaries of the original image
        new_x1 = max(0, min(width, new_x1))
        new_y1 = max(0, min(height, new_y1))
        new_x2 = max(0, min(width, new_x2))
        new_y2 = max(0, min(height, new_y2))
        # Append the transformed bounding box to the list
        transformed_bboxes.append([new_x1, new_y1, new_x2, new_y2, conf])
    # Return the list of transformed bounding boxes
    return transformed_bboxes


def process_rotation(
        image: np.ndarray,
        angle: float,
        detection_model: Yolov9,
        ocr_model: OcrModel,
        conf_thresh: float,
        iou_thresh: float,
) -> Tuple[List[List[float]], List[str], float]:
    """
    Process a single rotation angle: rotate image, detect barcodes, and transform bounding boxes.

    Args:
        image (np.ndarray): The original image to process.
        angle (float): The rotation angle to apply.
        detection_model (Yolov9): The barcode detection model.
        ocr_model (OcrModel): The OCR model for text recognition.

    Returns:
        Tuple[List[List[float]], List[str], float]: A tuple containing transformed bounding boxes,
        recognized texts, and average confidence score.
    """
    # Rotate the image and get the rotation matrix
    rotated_image, M = rotate_image(image, angle)
    # Compute the inverse of the rotation matrix
    Minv = invert_transformation_matrix(M)
    # Detect barcodes in the rotated image using the detection model
    bboxes = detection_model.detect_barcodes(rotated_image, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # Transform the bounding boxes back to the original image coordinates
    transformed_bboxes = transform_bounding_boxes(bboxes, Minv, image.shape)
    # Recognize text within the detected barcodes using the OCR model
    results = ocr_model.recognize(rotated_image, bboxes)
    # Extract the recognized texts from the results
    texts = [result[0] for result in results]
    # Extract the confidence scores of the recognized texts
    text_confs = [result[1] for result in results]
    # Calculate the average confidence score for the current rotation
    if bboxes:
        # If there are bounding boxes, compute the mean of detection and OCR confidences
        avg_confidence = np.mean([bbox[4] for bbox in bboxes] + text_confs)
    else:
        # If no bounding boxes are detected, set average confidence to zero
        avg_confidence = 0
    # Return the transformed bounding boxes, recognized texts, and average confidence
    return transformed_bboxes, texts, avg_confidence


def test_time_aug_inference(
        image: np.ndarray,
        rotations: List[float],
        detection_model: Yolov9,
        ocr_model: OcrModel,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45,
) -> Tuple[List[List[float]], List[str]]:
    """
    Process multiple rotations and return the best results based on confidence.

    Args:
        image (np.ndarray): The original image to process.
        rotations (List[float]): A list of rotation angles to apply.
        detection_model (Yolov9): The barcode detection model.
        ocr_model (OcrModel): The OCR model for text recognition.

    Returns:
        Tuple[List[List[float]], List[str]]: The best bounding boxes and recognized texts.
    """
    # Initialize variables to store the best results
    best_bboxes = None  # To store the best bounding boxes
    best_texts = None  # To store the best recognized texts
    best_confidence = -1  # Initialize best confidence to -1 (lower than any possible confidence)
    # Iterate over each rotation angle
    for angle in rotations:
        # Process the image for the current rotation angle
        transformed_bboxes, texts, avg_confidence = process_rotation(
            image, angle, detection_model, ocr_model, conf_thresh, iou_thresh
        )
        # If the average confidence of the current rotation is higher than the best so far
        if avg_confidence > best_confidence:
            # Update the best confidence score
            best_confidence = avg_confidence
            # Update the best bounding boxes
            best_bboxes = transformed_bboxes
            # Update the best recognized texts
            best_texts = texts
    # Return the best bounding boxes and texts found across all rotations
    return best_bboxes, best_texts
