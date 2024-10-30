# Project Title: Barcode detection and recognition using Deep Learning models

## Overview

This project is designed to send images from a local directory to a remote server, where they are processed to detect and recognize content within the images. The results are then displayed with bounding boxes and labels showing recognized text and confidence levels.

## Prerequisites
Before running the script, ensure you have the following installed:
- Python 3.9 or higher
- OpenCV-Python
- Requests library

You can install the required libraries using pip:
```bash
pip3 install opencv-python requests
```

## Running the Script

To run the script in PyCharm, follow these steps:

1. Open the project in PyCharm.
2. Navigate to the script `predict.py`.
3. Right-click on the file in the project explorer and choose `Run 'predict'`.

To run the script from the command line:

```bash
python3 predict.py
```

## Expected Output

The script will process each image in the specified directory, sending it to the server and then displaying each image with bounding boxes and recognized text overlays. Press any key to move to the next image.
