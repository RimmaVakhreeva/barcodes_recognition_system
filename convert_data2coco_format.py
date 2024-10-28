import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

# Set the base directory for images
images_path = Path("./")
# Load annotations from a tab-separated values file
data = pd.read_csv('./annotations.tsv', sep='\t')

# Create a temporary DataFrame to process only needed columns
temp_data = data[['filename', 'width', 'height']]
images = []
filenames_to_idx = {}

# Read images, store properties and create an index for each image
for idx, (_, (filename, w, h)) in enumerate(temp_data.iterrows()):
    filenames_to_idx[filename] = idx
    image = cv2.imread(str(images_path / filename))
    h, w, _ = image.shape  # Re-read dimensions in case of image discrepancies
    images.append({
        'id': idx,
        'width': w,
        'height': h,
        'file_name': filename
    })

# Create a dictionary mapping filenames to indices
annotations_id = []
for idx in enumerate(data['filename']):
    annotations_id.append(idx)
dict_annotations_id = dict(annotations_id)

# Define a single category of annotations (for simplicity, assuming a single object class)
categories = [{"id": 0, "name": 'item'}]

# Prepare annotations and track number of annotations per image
annotations, annotation_by_image_id = [], defaultdict(lambda: 0)
for idx, (_, row) in enumerate(data.iterrows()):
    current_filename = row['filename']
    image_index = filenames_to_idx[current_filename]
    y, x, height, width = row[['x_from', 'y_from', 'width', 'height']]
    annotation_by_image_id[image_index] += 1

    annotations.append({
        'id': idx,
        'image_id': image_index,
        'category_id': 0,
        "area": width * height,
        "bbox": [x, y, width, height],
        "iscrowd": 0
    })

# Filter images to include only those with annotations
filtered_images = [
    img
    for img in images
    if img['id'] in annotation_by_image_id and annotation_by_image_id[img['id']] > 0
]

# Shuffle the filtered images and split into training and testing sets
random.shuffle(filtered_images)
train_images, test_images = filtered_images[:int(len(filtered_images) * 0.8)], filtered_images[
                                                                               int(len(filtered_images) * 0.8):]
train_images_ids, test_images_ids = set([img["id"] for img in train_images]), set([img["id"] for img in test_images])
train_annotations = [ann for ann in annotations if ann["image_id"] in train_images_ids]
test_annotations = [ann for ann in annotations if ann["image_id"] in test_images_ids]

# Prepare directories for train and test datasets
train_folder = images_path / 'train2017'
train_folder.mkdir(parents=True, exist_ok=True)
test_folder = images_path / 'test2017'
test_folder.mkdir(parents=True, exist_ok=True)

# Move images to their respective folders
train_path_list = [Path(image['file_name']) for image in train_images]
test_path_list = [Path(image['file_name']) for image in test_images]
for image in train_path_list:
    shutil.move(str(image), train_folder / image.name)
for image in test_path_list:
    shutil.move(str(image), test_folder / image.name)

# Create a validation folder by copying the test folder
val_folder = images_path / 'val2017'
shutil.copytree(test_folder, val_folder, dirs_exist_ok=True)

# Write train, test, and validation data to JSON files
with open('./train.json', 'w') as f:
    json.dump({
        "images": train_images,
        "annotations": train_annotations,
        'categories': categories
    }, f)
with open('./test.json', 'w') as f:
    json.dump({
        "images": test_images,
        "annotations": test_annotations,
        'categories': categories
    }, f)
with open('./val.json', 'w') as f:
    json.dump({
        "images": test_images,
        "annotations": test_annotations,
        'categories': categories
    }, f)
