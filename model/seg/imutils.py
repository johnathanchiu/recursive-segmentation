from typing import Tuple
import numpy as np
from PIL import Image


def vstrips_process(
    img: Image.Image | np.ndarray, bbox: list[float, float, float, float], pad: int = 5
) -> Tuple[list[np.ndarray], list[bool]]:
    if isinstance(img, Image.Image):
        img = np.array(img)

    strips, labels = [], []
    for idx in range(0, img.shape[1], pad * 2):
        # if this chunk overlaps with the boundary of the bbox then the model should predict a segmentation here
        if (
            max(idx - pad, 0) < bbox[0] < idx + pad
            or max(idx - pad, 0) < bbox[0] + bbox[2] < idx + pad
        ):
            labels.append(1)
            # if debug:
            #     strips.append(img[:, max(idx - pad, 0) : idx + pad])
        else:
            labels.append(0)
            # if debug:
            #     size = idx + pad - max(idx - pad, 0)
            #     strips.append(np.ones((img.shape[0], size, 3)) * 0)
        strips.append(img[:, max(idx - pad, 0) : idx + pad])
    return strips, labels


def hstrips_process(
    img: Image.Image | np.ndarray, bbox: list[float, float, float, float], pad: int = 5
) -> Tuple[list[np.ndarray], list[bool]]:
    if isinstance(img, Image.Image):
        img = np.array(img)

    # If set to debug, strips returns a list of
    strips, labels = [], []
    for idx in range(0, img.shape[0], pad * 2):
        # if this chunk overlaps with the boundary of the bbox then the model should predict a segmentation here
        if (
            max(idx - pad, 0) < bbox[1] < idx + pad
            or max(idx - pad, 0) < bbox[1] + bbox[3] < idx + pad
        ):
            labels.append(1)
            # if debug:
            #     strips.append(img[max(idx - pad, 0) : idx + pad])
        else:
            labels.append(0)
            # size = idx + pad - max(idx - pad, 0)
            # strips.append(np.ones((size, img.shape[1], 3)) * 0)
        strips.append(img[max(idx - pad, 0) : idx + pad])
    return strips, labels


# def debug_strips(img: np.ndarray):
#     vstrips = vstrips_intersects(np.array(img), bbox_shifted)
#     hstrips = hstrips_intersects(np.array(img), bbox_shifted)
#     plt.imshow(np.hstack(vstrips).astype(np.uint8))
#     plt.imshow(np.vstack(hstrips).astype(np.uint8))
