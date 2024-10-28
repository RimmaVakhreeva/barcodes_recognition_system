from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

# Import model classes for barcode detection and OCR
from detection.detect import Yolov9
from ocr.model import OcrModel
from ocr.ocr_train_project.crnn import TransformerOcr

# Define the path where images are stored
IMAGES_PATH = Path("./images/test2017")

# Load the YoloV9 detection model with pre-trained weights
detection_model = Yolov9(weights=Path("./best.pt"))
# Load the OCR model with pre-trained weights
ocr_model = OcrModel(weights=Path("./crnn_last.pt"))


if __name__ == "__main__":
    # Iterate over all files in the specified directory
    for path in IMAGES_PATH.iterdir():
        # Read the image file to display it later
        image = np.array(Image.open(path)).astype(np.uint8)

        # Use the detection model to find barcodes in the image
        bboxes = detection_model.detect_barcodes(image)
        # Use the OCR model to recognize text within the detected barcodes
        texts = ocr_model.recognize(image, bboxes)

        # Draw bounding boxes and text on the image
        for (x1, y1, x2, y2, conf), text in zip(bboxes, texts):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image,
                        f'text: `{text}`', (int(x1), int(y1) - 2),
                        0, 1, (0, 0, 255),
                        thickness=4, lineType=cv2.LINE_AA)

        # Display the image
        cv2.imshow("image", image)
        # Wait for a short period before displaying the next image
        cv2.waitKey(0)
