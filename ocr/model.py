from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from ocr.ocr_train_project.crnn import TransformerOcr


class OcrModel:
    """
    OCR model that uses a CRNN architecture to recognize text in specified regions of an image.

    Attributes:
        weights (Path): Path to the pretrained model weights.
    """

    def __init__(self, weights: Path, device="cpu"):
        # Check if the provided weights file exists
        assert weights.exists()
        # Initialize the CRNN model with specified parameters
        self.model = TransformerOcr(
            cnn_backbone_name='resnet18d',
            cnn_backbone_pretrained=True,
            cnn_output_size=4608,
            transformer_features_num=128,
            transformer_dropout=0.1,
            transformer_nhead=32,
            transformer_num_layers=2,
            num_classes=12
        ).to(device)
        # Set fixed dimensions for the image crop
        self.h, self.w = 280, 523
        # Load the model weights
        self.model.load_state_dict(torch.load(str(weights), map_location='cpu'))
        # Determine the computing device (CUDA if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Define a vocabulary and a mapping from indices to characters
        self.vocab = '0123456789'
        self.index2char = {idx + 1: char for idx, char in enumerate(self.vocab)}
        self.index2char[0] = " "  # Add a space character for the blank class

    def recognize(self, image: np.ndarray, bboxes: List[np.ndarray]) -> List[str]:
        """
        Recognizes text within given bounding boxes of an image.

        Args:
            image (np.ndarray): The image to process.
            bboxes (List[np.ndarray]): List of bounding boxes with confidence scores.

        Returns:
            List[str]: The recognized text from each bounding box.
        """
        texts = []
        # Process each bounding box
        for (x1, y1, x2, y2, conf) in bboxes:
            # Crop, resize and normalize the image region defined by the bounding box
            crop = Image.fromarray(image[int(y1):int(y2), int(x1):int(x2)])
            crop = np.array(crop.resize((self.w, self.h)), dtype=np.uint8)
            crop = np.ascontiguousarray(crop.transpose((2, 0, 1))[::-1] / 255.0)[None, ...]
            # Predict the text using the CRNN model
            pred = self.model(torch.tensor(crop, dtype=torch.float32, device=self.device))
            text = self.model.decode_output(pred, vocab=self.vocab)[0]
            texts.append(text)
        return texts


if __name__ == "__main__":
    import cv2
    # Load an image and define a bounding box for testing the OCR model
    image_path = Path(
        "../images/test2017/0f61e632-59a2-4bc8-9119-24178ca64752--ru.8487fa87-b8cd-4def-9e54-5cf42bc4148e.jpg")
    image = cv2.imread(str(image_path))
    bboxes = [np.array([116.0, 499.0, 773.0, 765.0, 1.0])]

    # Initialize the OCR model and perform recognition
    model = OcrModel(Path("../crnn_last.pt"))
    print(model.recognize(image, bboxes))


# from pathlib import Path
# from typing import List
#
# import cv2
# import numpy as np
# import torch
#
# from ocr.ocr_train_project.crnn import CRNN
#
#
# class OcrModel:
#     def __init__(self, weights: Path):
#         assert weights.exists()
#         self.model = CRNN(
#             cnn_backbone_name='resnet18d',
#             cnn_backbone_pretrained=False,
#             cnn_output_size=4608,
#             rnn_features_num=128,
#             rnn_dropout=0.1,
#             rnn_bidirectional=True,
#             rnn_num_layers=2,
#             num_classes=11
#         )
#         self.h, self.w = 280, 523
#         self.model.load_state_dict(torch.load(str(weights), map_location='cpu'))
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.vocab = '0123456789'
#         self.index2char = {idx + 1: char for idx, char in enumerate(self.vocab)}
#         self.index2char[0] = " "
#
#     def recognize(self, image: np.ndarray, bboxes: List[np.ndarray]) -> List[str]:
#         texts = []
#         for (x1, y1, x2, y2, conf) in bboxes:
#             crop = image[int(y1):int(y2), int(x1):int(x2)]
#             crop = cv2.resize(crop, (self.w, self.h))
#             crop = np.ascontiguousarray(crop.transpose((2, 0, 1))[::-1] / 255.0)[None, ...]
#             pred = self.model(torch.tensor(crop, dtype=torch.float32, device=self.device))
#             text = self.model.decode_output(pred, vocab=self.vocab)[0]
#             texts.append(text)
#         return texts
#
#
# if __name__ == "__main__":
#     image_path = Path("/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/"
#                       "images/test2017/0f61e632-59a2-4bc8-9119-24178ca64752--ru.8487fa87-b8cd-4def-9e54-5cf42bc4148e.jpg")
#     image = cv2.imread(str(image_path))
#     bboxes = [np.array([116.0, 499.0, 773.0, 765.0, 1.0])]
#
#     model = OcrModel(Path("/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/crnn_best.pt"))
#     print(model.recognize(image, bboxes))