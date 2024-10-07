from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from .div import div_intersections


@dataclass
class ImageSection:
    bounding_box: Tuple[int, int, int, int]
    page_image: Image.Image
    vertical_seg: bool


def image_page_scan(
    image_crop: np.array,
    line_spacing: float = 5.0,
    vertical_scan: bool = True,
    pixel_crop_size: int = 5,
    debug: bool = False,
):
    shape = image_crop.shape[0] if vertical_scan else image_crop.shape[1]

    scan_intersects = []
    scan_lines = list(np.arange(0, shape, line_spacing))
    for scan_line in scan_lines:
        pixel_scan = (
            image_crop[
                max(int(scan_line) - pixel_crop_size, 0) : int(scan_line)
                + pixel_crop_size
            ]
            if vertical_scan
            else image_crop[
                :,
                max(int(scan_line) - pixel_crop_size, 0) : int(scan_line)
                + pixel_crop_size,
            ]
        )

        # Simple thresholding like this would result in incorrect results (e.g., if the background is not white)
        if np.mean(pixel_scan) < 254.8:
            scan_intersects.append(True)
            continue

        scan_intersects.append(False)

    debug_info = None
    if debug:
        debug_info = zip(scan_intersects, scan_lines)

    return div_intersections(scan_intersects, scan_lines), debug_info


def partition_image(page_section: ImageSection) -> List[Image.Image]:
    page_image = page_section.page_image
    vseg = page_section.vertical_seg

    page_breaks, _ = image_page_scan(
        np.array(page_section.page_image), line_spacing=10.0, vertical_scan=vseg
    )

    page_crops = []
    for section in page_breaks:
        p0, p1 = section
        if vseg:
            bbox = page_section.bounding_box
            crop = page_image.crop([0, p0, page_image.width, p1])
            # bug would be here, does scan line start from bottom or top
            bbox = [bbox[0], bbox[1] + p0, bbox[2], bbox[1] + p1]
        else:
            bbox = page_section.bounding_box
            crop = page_image.crop([p0, 0, p1, page_image.height])
            bbox = [bbox[0] + p0, bbox[1], bbox[0] + p1, bbox[3]]
        page_crops.append(
            ImageSection(bounding_box=bbox, page_image=crop, vertical_seg=not vseg)
        )

    return page_crops
