import os
from glob import glob
from typing import Callable, List, Optional, Tuple, Union

import albumentations as alb
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image


def render_text(
        text: str,
        font: Union[ImageFont.FreeTypeFont, Callable[[], ImageFont.FreeTypeFont]],
        color: Union[Tuple[int, int, int], Callable[[], Tuple[int, int, int]]] = (0, 0, 0),
        spacing: Union[float, Callable[[], float]] = 0.7,
        sp_var: Union[float, Callable[[], float]] = 0.05,
        alpha: Union[int, Callable[[], int]] = 255,
        pad: int = 1,
        transforms: Optional[alb.BasicTransform] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate random spacings for each character in the text
    spacings = np.random.uniform(
        spacing - sp_var * font.size,
        spacing + sp_var * font.size,
        size=len(text),
    )
    # Determine the overall width and height of the text image
    text_w = int(sum(font.getsize(c)[0] for c in text) + sum(spacings[:-1]))
    text_h = int(font.getsize(text)[1])

    # Create a new image with transparent background
    image = Image.new(
        'RGBA',
        (text_w + 2 * pad, text_h + 2 * pad),
        tuple(list((0, 0, 0)) + [0]),
    )
    draw = ImageDraw.Draw(image)

    # Draw each character on the image at calculated positions
    x = pad
    for c, sp in zip(text, spacings):
        w, _ = font.getsize(c)
        draw.text((x, pad), c, tuple(color), font=font)
        x += w + sp

    # Convert the image to a numpy array
    image = np.asarray(image).astype(np.float32)

    # Create a mask from the alpha channel
    mask = image[..., 3]
    mask = mask / max(mask.max(), 1) * alpha
    mask = mask.astype(np.uint8)
    # Apply transformations to the mask if any
    mask = transforms(image=mask)['image'] if transforms else mask

    # Convert the RGBA image to RGB
    image = image[..., :3].astype(np.uint8)
    return image, mask


def load_fonts(fonts_dir: str, sizes: List[int]) -> Optional[List[ImageFont.FreeTypeFont]]:
    # Find all font files in the specified directory
    to_fonts = glob(os.path.join(fonts_dir, '*'))
    # Repeat sizes for each font and tile fonts for each size
    sizes = np.repeat(sizes, len(to_fonts))
    to_fonts = np.tile(to_fonts, len(sizes))
    # Load each font at each specified size
    return [ImageFont.truetype(font, size) for font, size in zip(to_fonts, sizes)]


def overlay(
        back_img,
        front_img,
        front_mask,
        pad: Union[Tuple[int, int, int, int], Callable[[], Tuple[int, int, int, int]]] = (0, 0, 0, 0),
        transforms: Optional[alb.BasicTransform] = None,
        alpha: Union[int, Callable[[], int]] = 255,
        fit_mode: Optional[str] = 'background',
) -> Tuple[np.ndarray, np.ndarray]:
    # Extract padding values
    lpad, tpad, rpad, bpad = pad

    # Get dimensions of the background and foreground images
    bh, bw = back_img.shape[:2]
    fh, fw = front_img.shape[:2]

    # Resize or reposition images based on fit mode
    if fit_mode is None:
        assert fw + lpad + rpad <= bw and fh + tpad + bpad <= bh, 'Foreground does not fit within the specified paddings'
        # Randomly position the foreground image within the background
        x_left = np.random.randint(0, bw - fw - lpad - rpad)
        y_left = np.random.randint(0, bh - fh - tpad - bpad)
        x_right = x_left + lpad + fw + rpad
        y_right = y_left + tpad + fh + bpad
        back_img = back_img[y_left:y_right, x_left:x_right, ...]
    if fit_mode == 'background':
        # Resize the background image to fit the foreground
        back_img = cv2.resize(back_img, (fw + rpad + lpad, fh + tpad + bpad))
    if fit_mode == 'foreground':
        # Resize the foreground image and mask to fit the background
        front_img = cv2.resize(front_img, (bw, bh))
        front_mask = cv2.resize(front_mask, (bw, bh))

    # Apply the foreground image onto the background image using the mask
    front_mask_c = np.expand_dims(front_mask, -1) / 255.0
    back_img[
    max(0, tpad): tpad + fh, max(0, lpad): lpad + fw, :
    ] = front_img * front_mask_c + back_img[
                                   max(0, tpad): tpad + fh, max(0, lpad): lpad + fw, :
                                   ] * (1 - front_mask_c)

    # Apply transformations to the composited image if any
    return transforms(image=back_img)['image'] if transforms else back_img
