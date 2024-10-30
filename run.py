# Import necessary libraries
import io
from pathlib import Path

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


# Define a POST endpoint for processing uploaded images
@app.post("/scan/")
async def scan_barcodes(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    image_data = await file.read()
    # Convert the file to an image array
    image = np.array(Image.open(io.BytesIO(image_data)))

    # Use the detection model to find barcodes in the image
    bboxes = detection_model.detect_barcodes(image)
    # Use the OCR model to recognize text within the detected barcodes
    texts = ocr_model.recognize(image, bboxes)

    # Compile the detection and recognition results into a list
    response_data = []
    for bbox, text in zip(bboxes, texts):
        response_data.append({
            "bbox": bbox[:4].tolist(),
            "bbox_confidence": bbox[4].tolist(),
            "text": text
        })

    # Return the results as a JSON response
    return JSONResponse(content=response_data)


# Main block to run the server if this script is executed directly
if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app with Uvicorn on localhost at port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
