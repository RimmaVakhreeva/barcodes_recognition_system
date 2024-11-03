# Import necessary libraries
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Import model classes for barcode detection and OCR
from detection.detect import Yolov9
from ocr.model import OcrModel

# Initialize the FastAPI application
app = FastAPI()
# Load the YoloV9 detection model with pre-trained weights
detection_model = Yolov9(weights=Path("./best.pt"))
# Load the OCR model with pre-trained weights
ocr_model = OcrModel(weights=Path("./crnn_last.pt"))


def rotate_bbox(bbox, angle, original_shape, rotated_shape):
    """Rotate bounding box coordinates to match the original image orientation."""
    orig_h, orig_w = original_shape[:2]
    rot_h, rot_w = rotated_shape[:2]

    # Center of the rotated image
    rot_cx, rot_cy = rot_w / 2, rot_h / 2

    # Center of the original image
    orig_cx, orig_cy = orig_w / 2, orig_h / 2

    angle_rad = np.deg2rad(angle)

    # Calculate new bounding box coordinates
    x1, y1, x2, y2 = bbox[:4]
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # Translate points to the origin (center of the rotated image)
    translated_points = points - [rot_cx, rot_cy]

    # Rotate points
    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_points = np.dot(translated_points, rotation_matrix)

    # Scale points to match the original image size
    scale_x = orig_w / rot_w
    scale_y = orig_h / rot_h
    scaled_points = rotated_points * [scale_x, scale_y]

    # Translate points back (to the center of the original image)
    final_points = scaled_points + [orig_cx, orig_cy]

    # Get the new bounding box coordinates
    new_x1, new_y1 = np.min(final_points, axis=0)
    new_x2, new_y2 = np.max(final_points, axis=0)

    # Ensure the coordinates are within the original image boundaries
    new_x1 = max(0, min(orig_w, new_x1))
    new_y1 = max(0, min(orig_h, new_y1))
    new_x2 = max(0, min(orig_w, new_x2))
    new_y2 = max(0, min(orig_h, new_y2))

    return [new_x1, new_y1, new_x2, new_y2, bbox[4]]  # Keep the confidence score


# Define a POST endpoint for processing uploaded images
@app.post("/scan/")
async def scan_barcodes(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    image_data = await file.read()
    # Convert the file to an image array
    image = np.array(Image.open(io.BytesIO(image_data)))

    # Initialize variables to store the best results
    best_bboxes = None
    best_texts = None
    best_confidence = -1

    # Define the rotations to apply
    rotations = [0, 90, 180, 270]

    for angle in rotations:
        # Rotate the image
        rotated_image = np.array(Image.fromarray(image).rotate(angle, expand=True))

        # Use the detection model to find barcodes in the rotated image
        bboxes = detection_model.detect_barcodes(rotated_image)
        # Rotate bounding boxes to match the original image orientation
        rotated_bboxes = [rotate_bbox(bbox, angle, image.shape, rotated_image.shape) for bbox in bboxes]
        # Use the OCR model to recognize text within the detected barcodes
        results = ocr_model.recognize(rotated_image, bboxes)
        texts = [result[0] for result in results]
        text_confs = [result[1] for result in results]


        # for ((x1, y1, x2, y2, conf), text, text_conf) in zip(bboxes, texts, text_confs):
        #     x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        #     cv2.rectangle(rotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
        #     cv2.putText(rotated_image,
        #                 f'conf: {conf:.2f}, text: `{text}`',
        #                 (x1, y1 - 2), 0, 1, (0, 255, 0),
        #                 thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow("image", rotated_image)
        # cv2.waitKey(0)



        # Calculate the average confidence for the current rotation
        avg_confidence = np.mean([bbox[4] for bbox in bboxes] + text_confs)

        # Update the best results if the current confidence is higher
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_bboxes = rotated_bboxes
            best_texts = texts

    # Compile the detection and recognition results into a list
    response_data = []
    for bbox, text in zip(best_bboxes, best_texts):
        response_data.append({
            "bbox": list(map(float, bbox[:4])),
            "bbox_confidence": float(bbox[4]),
            "text": text
        })

    # Return the results as a JSON response
    return JSONResponse(content=response_data)


# Main block to run the server if this script is executed directly
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app with Uvicorn on localhost at port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
