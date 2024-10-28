import random
from pathlib import Path

import albumentations as alb
import barcode
import cv2
import numpy as np
import torch
from barcode.writer import ImageWriter

from ocr.ocr_train_project.generator.src.generator import overlay


class BarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, vocab, real_ds_path: Path):
        self.epoch_size = epoch_size
        self.char2index = dict((char, idx + 1) for idx, char in enumerate(vocab))
        self.pad_value = 0
        self.eos_value = len(vocab) + 1
        self.h, self.w = 280, 523

        background_parent_path = Path(__file__).parent / "backgrounds"
        self.reference_images = [
            path
            for path in (background_parent_path.parent.parent / "test_images").iterdir()
        ]
        self.backgrounds = [
            background_parent_path / "background_0.jpg",
            background_parent_path / "background_1.jpg",
            background_parent_path / "background_2.jpg",
        ]
        self.real_ds_paths = list(real_ds_path.iterdir())

    def _synthetic_generate(self):
        value = str(np.random.randint(10e11, 10e12))
        ean = barcode.EAN13(value, writer=ImageWriter(), guardbar=True)
        value_w_checksum = ean.get_fullcode()
        value_w_checksum = value_w_checksum.replace(" ", "")
        rendered_barcode = ean.render()
        rendered_barcode = 255 - cv2.cvtColor(np.asarray(rendered_barcode), cv2.COLOR_RGB2GRAY)
        h, w = rendered_barcode.shape
        barcode_image = np.zeros((h, w, 3), dtype=np.uint8)
        background = cv2.imread(str(random.choice(self.backgrounds)))
        barcode_img = overlay(
            background,
            barcode_image,
            rendered_barcode,
            transforms=alb.Compose([
                alb.ShiftScaleRotate(shift_limit=0.005, scale_limit=(-0.1, 0.1), rotate_limit=(-13, 13),
                                     border_mode=cv2.BORDER_CONSTANT, p=1),
                alb.GaussNoise(p=0.3),
                alb.OneOf([
                    alb.MotionBlur(blur_limit=11, p=0.4),
                    alb.MedianBlur(blur_limit=7, p=0.4),
                    alb.Blur(blur_limit=11, p=0.4),
                ], p=0.5),
                alb.OneOf([
                    alb.OpticalDistortion(p=0.4, border_mode=cv2.BORDER_CONSTANT),
                    alb.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT),
                ], p=0.3),
                alb.OneOf([
                    alb.CLAHE(clip_limit=2),
                    alb.RandomBrightnessContrast(),
                ], p=0.3),
                alb.HueSaturationValue(p=0.3),
                alb.RGBShift(r_shift_limit=(-20, 20), p=0.3),
                alb.ColorJitter(brightness=(0.6, 1),
                                contrast=(0.6, 1),
                                saturation=(0.6, 1),
                                hue=(-0.5, 0.5), p=0.3),
                alb.FDA(self.reference_images, p=0.6)
            ])
        )
        return barcode_img, value_w_checksum

    def _load_real_dataset(self):
        path = random.choice(self.real_ds_paths)
        image = cv2.imread(str(path))
        assert image is not None
        text = path.stem
        transforms = alb.Compose([
            alb.ShiftScaleRotate(shift_limit=0.001, scale_limit=(-0.01, 0.01), rotate_limit=0,
                                 border_mode=cv2.BORDER_CONSTANT, p=1),
            alb.GaussNoise(p=0.2),
            alb.OneOf([
                alb.MotionBlur(blur_limit=7, p=0.4),
                alb.Blur(blur_limit=7, p=0.4),
            ], p=0.3),
            alb.OneOf([
                alb.OpticalDistortion(p=0.4, border_mode=cv2.BORDER_CONSTANT),
                alb.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT),
            ], p=0.2),
            alb.OneOf([
                alb.CLAHE(clip_limit=2),
                alb.RandomBrightnessContrast(),
            ], p=0.2),
            alb.HueSaturationValue(p=0.3),
            alb.RGBShift(r_shift_limit=(-20, 20), p=0.3),
            alb.ColorJitter(brightness=(0.6, 1),
                            contrast=(0.6, 1),
                            saturation=(0.6, 1),
                            hue=(-0.5, 0.5), p=0.2),
            alb.FDA(self.reference_images, p=0.4),
            alb.Resize(height=self.h, width=self.w)
        ])
        return transforms(image=image)['image'], text

    def __getitem__(self, i: int):
        if random.random() > 0.9:
            img, text = self._synthetic_generate()
        else:
            img, text = self._load_real_dataset()

        # cv2.imshow("barcode_img", img)
        # cv2.waitKey(0)

        img = torch.FloatTensor(img / 255.0).permute(2, 0, 1)
        classes = self.encode_value(text)
        classes = classes + [self.eos_value] + [self.pad_value] * (17 - (len(classes) + 1))
        return (
            img,
            torch.LongTensor(classes),
            text
        )

    def __len__(self):
        return self.epoch_size

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)


class TestBarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, directory, vocab):
        self.directory = Path(directory)
        self.image_files = list(self.directory.glob('*.jpg'))
        self.h, self.w = 280, 523
        self.vocab = vocab
        self.char2index = dict((char, vocab.index(char) + 1) for char in vocab)
        self.pad_value = 0
        self.eos_value = len(vocab) + 1

    def __getitem__(self, i: int):
        img_path = self.image_files[i]
        image = cv2.imread(str(img_path))
        image = cv2.resize(image, (self.w, self.h))
        image = torch.FloatTensor(image / 255.0).permute(2, 0, 1)
        classes = self.encode_value(img_path.stem)
        classes = classes + [self.eos_value] + [self.pad_value] * (17 - (len(classes) + 1))
        return (
            image,
            torch.LongTensor(classes),
            img_path.stem
        )

    def __len__(self):
        return len(self.image_files)

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)


if __name__ == "__main__":
    ds = BarcodeDataset(epoch_size=1024, vocab="0123456789", real_ds_path=Path(__file__).parent.parent / "train_images")
    for idx in range(1024):
        _ = ds[idx]
