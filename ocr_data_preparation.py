from pathlib import Path

import pandas as pd


def create_ocr_annotations():
    """
    Process annotations from a TSV file to create a simplified CSV file
    with only relevant information for OCR tasks, including filenames and bounding box coordinates.

    Returns:
        None
    """

    # Set the base directory where images are stored
    images_path = Path("./images")
    # Load annotations from a tab-separated values file
    data = pd.read_csv('./annotations.tsv', sep='\t')
    # Convert the 'filename' column to Path objects for easier file handling
    data['filename'] = data['filename'].apply(lambda x: Path(x))

    # Define the path to the folder containing test images
    test_folder = images_path / 'test2017'

    # Initialize lists to store data for each image
    test_images = []
    codes = []
    x_from_bbox = []
    y_from_bbox = []
    width_bbox = []
    height_bbox = []

    # Iterate through each row in the annotations data
    for index, row in data.iterrows():
        # Iterate through each file in the test images folder
        for img in test_folder.iterdir():
            # Check if the file extension is one of the specified image formats and the names match
            if img.suffix in ['.jpg', '.png', '.jpeg'] and row['filename'].name == img.name:
                # Store the string representation of the image path
                test_images.append(str(img))
                # Store the code associated with the annotation
                codes.append(row['code'])
                # Store the bounding box coordinates
                x_from_bbox.append(row['x_from'])
                y_from_bbox.append(row['y_from'])
                width_bbox.append(row['width'])
                height_bbox.append(row['height'])

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'filename': test_images,
        'code': codes,
        'x_from': x_from_bbox,
        'y_from': y_from_bbox,
        'width': width_bbox,
        'height': height_bbox
    })

    # Write the DataFrame to a CSV file without index column
    df.to_csv('ocr_annotations.csv', index=False)


if __name__ == "__main__":
    # Execute the function to create OCR annotations
    create_ocr_annotations()
