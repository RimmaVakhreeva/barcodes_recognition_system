from pathlib import Path

import cv2
import requests

# Define the path where images are stored
IMAGES_PATH = Path("./images/test2017")
# URL of the server to which the images will be posted
#URL = "https://ec2-3-106-214-22.ap-southeast-2.compute.amazonaws.com:80/scan/"
URL = "http://ec2-3-106-214-22.ap-southeast-2.compute.amazonaws.com:80/scan/"

if __name__ == "__main__":
    # Iterate over all files in the specified directory
    for path in IMAGES_PATH.iterdir():
        with open(path, 'rb') as file:
            # Send POST request to the server with the image file
            response = requests.post(URL, files={'file': file})
            # Check if the request was successful
            if response.status_code != 200:
                # Print an error message if the request failed
                print(f"Request error ({response.status_code}): {response.reason}")
                continue

        # Read the image file to display it later
        image = cv2.imread(str(path))
        try:
            # Try to parse the JSON response from the server
            data = response.json()
        except Exception as e:
            # Skip the current iteration if there's an error parsing the JSON
            continue

        # Iterate over each detected object in the response
        for item in data:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, item["bbox"])
            # Extract confidence score of the bounding box
            conf = item["bbox_confidence"]
            # Extract recognized text
            text = item["text"]
            # Draw a rectangle around the detected object
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
            # Put text on the image showing confidence and recognized text
            cv2.putText(image,
                        f'conf: {conf:.2f}, text: `{text}`',
                        (x1, y1 - 2), 0, 1, (0, 0, 255),
                        thickness=4, lineType=cv2.LINE_AA)
        # Display the image
        cv2.imshow("image", image)
        # Wait for a short period before displaying the next image
        cv2.waitKey(0)
