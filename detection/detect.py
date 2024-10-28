import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

# Absolute path to the current file
FILE = Path(__file__).resolve()
# Root directory for YOLOv9, assuming it's a subdirectory named "yolov9"
ROOT = FILE.parents[0] / "yolov9"
# Add the YOLO root directory to the system path if it's not already there
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Update ROOT to be a relative path from the current working directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Import necessary functions and classes from other modules in the project
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2,
                           non_max_suppression, scale_boxes)


class Yolov9:
    """
    A class for YOLOv9 object detection. Initializes with model weights and device setting (CPU/GPU),
    sets various detection parameters, and includes a method to detect barcodes from images using the YOLOv9 model.

    Attributes:
        weights (Path): Path to the model weights file.
        device (str): Device to run the model on ('cpu' or 'gpu').
    """

    def __init__(self, weights: Path, device: str = "cpu"):
        # Ensure the weights file exists
        assert weights.exists()
        # Initialize the model backend with given weights and device
        self.model = DetectMultiBackend(weights, device=device)
        # Model stride used for image preprocessing
        self.model_stride = self.model.stride
        # Check if the model is a PyTorch model
        self.is_torch_model = self.model.pt
        # Default image source directory
        self.source = 'images/test2017'
        # Set the standard image size for processing
        self.img_size = (640, 640)
        # Confidence threshold for detections
        self.conf_thres = 0.5
        # Intersection over Union threshold for determining valid detections
        self.iou_thres = 0.45
        # Whether to apply class-agnostic non-max suppression
        self.agnostic_nms = False
        # Maximum number of detections to allow
        self.max_det = 1000

    def detect_barcodes(self, image: np.ndarray) -> List[np.ndarray]:
        # Preprocess the image for model input
        augmented_image = letterbox(image, self.img_size, stride=self.model_stride, auto=self.is_torch_model)[0]
        # Convert image from HWC to CHW format and BGR to RGB
        augmented_image = augmented_image.transpose((2, 0, 1))[::-1]
        # Ensure the image array is contiguous
        augmented_image = np.ascontiguousarray(augmented_image)

        # Convert image to PyTorch tensor and transfer to the specified device
        augmented_image = torch.from_numpy(augmented_image).to(self.model.device)
        # Adjust the image data type based on model precision requirements
        augmented_image = augmented_image.half() if self.model.fp16 else augmented_image.float()
        # Normalize the pixel values
        augmented_image /= 255
        # Add a batch dimension
        augmented_image = augmented_image[None, ...]

        # Perform the detection
        pred = self.model(augmented_image, augment=False, visualize=False)
        pred = pred[0][1]

        # Apply non-max suppression to filter detections
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            classes=None,
            agnostic=self.agnostic_nms,
            max_det=self.max_det
        )
        # Adjust bounding box dimensions to original image size and convert to CPU and numpy format
        for i, det in enumerate(pred):
            det[:, :4] = scale_boxes(augmented_image.shape[2:], det[:, :4], image.shape).round()
        return [bbox.cpu().numpy()[0, :5] for bbox in pred if len(bbox) > 0]


if __name__ == "__main__":
    # Path to the model weights and image directory
    weights = Path("../best.pt")
    images_path = Path("../images/test2017")

    # Initialize the YOLOv9 detector
    yolov9 = Yolov9(weights)
    # Loop through images in the specified directory
    for image_path in images_path.glob('*'):
        # Process only .jpg files
        if image_path.suffix.lower() in ['.jpg']:
            # Read the image
            image = cv2.imread(str(image_path))
            # Proceed if the image is valid
            if image is not None:
                # Detect barcodes in the image
                results = yolov9.detect_barcodes(image)
                # Draw bounding boxes on the detected barcodes and display the image
                for (x1, y1, x2, y2, conf) in results:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2,
                                  lineType=cv2.LINE_AA)
                    cv2.imshow("image", image)
                    cv2.waitKey(0)
            else:
                print(f"Failed to load {image_path.name}")
        else:
            continue
