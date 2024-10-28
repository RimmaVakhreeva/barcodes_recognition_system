from pathlib import Path

import cv2
import pandas as pd

# Define the base path for image storage
images_path = Path("./images")
# Define the path for storing cropped images
cropped_images_path = images_path / 'train_images'
cropped_images_path.mkdir(parents=True, exist_ok=True)
# Load annotations from a CSV file into a DataFrame
data = pd.read_csv('train_ocr_annotations.csv')

# Convert the 'filename' column to Path objects for easier file handling
data['filename'] = data['filename'].apply(lambda x: Path(x))

# Define the path to the folder containing test images
test_folder = images_path / 'train2017'

# Iterate through each row in the DataFrame
for index, row in data.iterrows():
    # Iterate through each file in the test images folder
    for img_path in test_folder.iterdir():
        # Check if the file extension is one of the specified image formats and the names match
        if img_path.suffix in ['.jpg', '.png', '.jpeg'] and row['filename'].name == img_path.name:
            # Read the image from file
            img = cv2.imread(str(img_path))
            # Assert to ensure the image is loaded correctly
            assert img is not None
            # Calculate the coordinates for cropping based on the DataFrame
            x1 = row['y_from']
            y1 = row['x_from']
            x2 = x1 + row['height']
            y2 = y1 + row['width']
            # Crop the image using calculated coordinates
            crop = img[int(y1):int(y2), int(x1):int(x2)]

            # Construct the path for the cropped image and save the file with a specific code as its filename
            crop_image_path = cropped_images_path / f"{row['code']}.jpg"
            cv2.imwrite(str(crop_image_path), crop)
