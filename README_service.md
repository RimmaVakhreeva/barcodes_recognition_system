# Image Processing Service

This service allows you to upload images and receive processed data that includes object detection bounding boxes, confidence scores, and recognized text within those bounding boxes.

## Service Overview

- **Endpoint URL**: `http://ec2-54-253-153-7.ap-southeast-2.compute.amazonaws.com:80/scan/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

The service accepts image files sent via a POST request and returns a JSON response containing detection results.

## Input Data

- **Image File**: The image should be sent as a file in the form data with the key `file`.
- **Supported Formats**: JPEG, PNG, and other common image formats.

### Example of Sending an Image

Using `requests` library in Python:

```python
import requests

url = "http://ec2-54-253-153-7.ap-southeast-2.compute.amazonaws.com:80/scan/"

with open('path_to_your_image.jpg', 'rb') as img_file:
    response = requests.post(url, files={'file': img_file})
```

## Output Data

The service returns a JSON array where each element represents a detected object within the image.

Each element in the array includes:

- **`bbox`**: A list of four integers `[x1, y1, x2, y2]` indicating the coordinates of the bounding box.
- **`bbox_confidence`**: A float between 0 and 1 representing the confidence level of the detection.
- **`text`**: A string containing the text recognized within the bounding box.

### Example JSON Response

```json
[
  {
    "bbox": [50, 100, 200, 300],
    "bbox_confidence": 0.85,
    "text": "Detected Text Here"
  },
  {
    "bbox": [250, 150, 400, 350],
    "bbox_confidence": 0.90,
    "text": "Another Text"
  }
]
```

## How to Use the Service

1. **Prepare Your Image**: Have the image file you want to process ready on your local machine.

2. **Send a POST Request**:

   - Use a tool like `curl`, `Postman`, or a programming language like Python to send a POST request.
   - The image file should be included in the form data with the key `file`.

   **Example using `curl`:**

   ```bash
   curl -X POST -F 'file=@/path/to/your/image.jpg' \
   'http://ec2-54-253-153-7.ap-southeast-2.compute.amazonaws.com:80/scan/'
   ```

   **Example using Python and `requests`:**

   ```python
   import requests

   url = "http://ec2-54-253-153-7.ap-southeast-2.compute.amazonaws.com:80/scan/"

   with open('path_to_your_image.jpg', 'rb') as img_file:
       response = requests.post(url, files={'file': img_file})

   if response.status_code == 200:
       data = response.json()
       print(data)
   else:
       print(f"Error: {response.status_code} - {response.reason}")
   ```

3. **Process the Response**:

   - If the request is successful (`status_code` 200), the response will be a JSON array containing the detection results.
   - You can parse this JSON to extract bounding boxes, confidence scores, and recognized text.

## Understanding the Output

- **Bounding Box (`bbox`)**:

  - Represents the area where an object or text was detected.
  - Coordinates are in the format `[x1, y1, x2, y2]`, where `(x1, y1)` is the top-left corner and `(x2, y2)` is the bottom-right corner.

- **Confidence Score (`bbox_confidence`)**:

  - Indicates how confident the model is about the detection.
  - Ranges from 0 (no confidence) to 1 (full confidence).

- **Recognized Text (`text`)**:

  - The text extracted from within the bounding box.
  - Useful for applications involving OCR (Optical Character Recognition).
