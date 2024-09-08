from typing import List

from pdfplumber.page import Page
import numpy as np

OBJECT_TYPES = ["line", "curve", "rect", "char", "image"]


def page_scan(
    page: Page,
    line_spacing: float = 5.0,
    vertical_scan: bool = True,
    debug: bool = False,
):
    page_objs = page.objects
    page_bbox = page.bbox
    # vertical scan implies the lines are going across the page dropped from top to bottom
    if not vertical_scan:
        page_dim = (page_bbox[1], page_bbox[3])
        p0, p1 = "x0", "x1"
    else:
        page_dim = (page_bbox[0], page_bbox[2])
        p0, p1 = "top", "bottom"

    scan_intersects = []
    scan_lines = list(np.arange(*page_dim, line_spacing))
    for scan_line in scan_lines:
        is_crossed = False
        for obj_type in OBJECT_TYPES:
            for obj in page_objs[obj_type]:
                if obj[p0] < scan_line < obj[p1]:
                    scan_intersects.append(True)
                    is_crossed = True
                    break
            if is_crossed:
                break

        if not is_crossed:
            scan_intersects.append(False)

    debug_info = None
    if debug:
        debug_info = zip(scan_intersects, scan_lines)

    return div_intersections(scan_intersects, scan_lines), debug_info


def div_intersections(intersections: List[bool], scan_lines: List[float]):
    """This function essentially splits a boolean list apart. For instance,
    `intersections` will be a boolean list. It will segement the consecutive sections
    of `True` groups. This will then be used to match with `scan_lines` to get
    the start and end sections of the page.
    """
    section_ints = []
    section_start = None

    for i, intersects in enumerate(intersections):
        if intersects:
            if section_start is None:
                section_start = i
        elif section_start is not None:
            section_ints.append((section_start, i - 1))
            section_start = None

    if section_start is not None:
        section_ints.append((section_start, len(intersections) - 1))

    section_crop_dims = []
    for section_int in section_ints:
        start, end = section_int
        p1 = scan_lines[min(end + 1, len(scan_lines) - 1)]
        p0 = scan_lines[max(start - 1, 0)]
        section_crop_dims.append((int(p0), int(p1)))

    return section_crop_dims


def image_scan(
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
