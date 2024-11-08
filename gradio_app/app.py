import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from detection.detect import Yolov9
from inference_funcs import test_time_aug_inference
from ocr.model import OcrModel

os.environ["GRADIO_TEMP_DIR"] = "./tmp"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize_bboxes(image, data):
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
                    (x1, y1 - 2), 0, 0.8, (0, 255, 0),
                    thickness=2, lineType=cv2.LINE_AA)
    return image


def recognize_image(input_img, conf_threshold, iou_threshold):
    image = np.array(input_img)
    # Define the list of rotation angles to apply to the image
    rotations = [0, 90, 180, 270]
    # Get the best detection and recognition results from the image across all rotations
    best_bboxes, best_texts = test_time_aug_inference(
        image,
        rotations,
        detection_model,
        ocr_model,
        conf_thresh=conf_threshold,
        iou_thresh=iou_threshold
    )
    # Initialize a list to compile the response data
    data = []
    # Check if any bounding boxes were detected
    if best_bboxes:
        # Iterate over each bounding box and its corresponding recognized text
        for bbox, text in zip(best_bboxes, best_texts):
            # Append the bounding box and text information to the response data
            data.append({
                "bbox": list(map(float, bbox[:4])),  # The bounding box coordinates
                "bbox_confidence": float(bbox[4]),  # The confidence score of the bounding box
                "text": text  # The recognized text within the bounding box
            })

    vis_result = visualize_bboxes(input_img, data)
    return vis_result


def gradio_reset():
    return gr.update(value=None), gr.update(value=None)


if __name__ == "__main__":
    root_path = os.path.abspath(os.getcwd())

    # Load the YoloV9 detection model with pre-trained weights
    detection_model = Yolov9(weights=Path("./best.pt"), device=device)

    # Load the OCR model with pre-trained weights
    ocr_model = OcrModel(weights=Path("./crnn_last.pt"), device=device)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label=" ", interactive=True)
                with gr.Row():
                    clear = gr.Button(value="Clear")
                    predict = gr.Button(value="Detect", interactive=True, variant="primary")

                with gr.Row():
                    conf_threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                    )

                with gr.Row():
                    iou_threshold = gr.Slider(
                        label="NMS IOU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                    )

                with gr.Accordion("Examples:"):
                    example_root = os.path.join(os.path.dirname(__file__), "assets", "example")
                    gr.Examples(
                        examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                  _.endswith("jpg")],
                        inputs=[input_img],
                    )
            with gr.Column():
                gr.Button(value="Predict Result:", interactive=False)
                output_img = gr.Image(label=" ", interactive=False)

        clear.click(gradio_reset, inputs=None, outputs=[input_img, output_img])
        predict.click(recognize_image, inputs=[input_img, conf_threshold, iou_threshold], outputs=[output_img])

    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
