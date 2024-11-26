# Import necessary libraries for handling files and paths
import io
from pathlib import Path

# Import OpenCV for image processing, NumPy for numerical operations
import numpy as np

# Import PIL for image handling
from PIL import Image

# Import FastAPI components for building the API
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Import model classes for barcode detection and OCR (assumed to be custom modules)
from detection.detect import Yolov9
from inference_funcs import test_time_aug_inference
from ocr.model import OcrModel

# Initialize the FastAPI application
app = FastAPI()

# Load the YoloV9 detection model with pre-trained weights
detection_model = Yolov9(weights=Path("./best.pt"))

# Load the OCR model with pre-trained weights
ocr_model = OcrModel(weights=Path("./crnn_best.pt"))


# Define a POST endpoint for processing uploaded images
@app.post("/")
async def scan_barcodes(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint to scan barcodes in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: A JSON response containing detected bounding boxes and recognized texts.
    """
    # Read the uploaded file's binary content into memory
    image_data = await file.read()
    # Open the image using PIL and convert it to a NumPy array
    image = np.array(Image.open(io.BytesIO(image_data)))
    # Define the list of rotation angles to apply to the image
    rotations = [0, 90, 180, 270]
    # Get the best detection and recognition results from the image across all rotations
    best_bboxes, best_texts = test_time_aug_inference(
        image, rotations, detection_model, ocr_model
    )
    # Initialize a list to compile the response data
    response_data = []
    # Check if any bounding boxes were detected
    if best_bboxes:
        # Iterate over each bounding box and its corresponding recognized text
        for bbox, text in zip(best_bboxes, best_texts):
            # Append the bounding box and text information to the response data
            response_data.append({
                "bbox": list(map(float, bbox[:4])),  # The bounding box coordinates
                "bbox_confidence": float(bbox[4]),  # The confidence score of the bounding box
                "text": text  # The recognized text within the bounding box
            })

    # Return the compiled results as a JSON response
    return JSONResponse(content=response_data)


# Main block to run the server if this script is executed directly
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app with Uvicorn on localhost at port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
