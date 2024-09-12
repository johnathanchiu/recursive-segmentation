from dataclasses import dataclass
from typing import List, Tuple

from pdfplumber.page import CroppedPage, Page
import numpy as np

from .div import div_intersections


OBJECT_TYPES = ["line", "curve", "rect", "char", "image"]


@dataclass
class Section:
    page_crop: CroppedPage
    vertical_seg: bool
    seg_depth: int = 0


def pdf_page_scan(
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


def section_page(
    page: Section,
    page_breaks: List[Tuple[int, int]],
    vertical_div: bool = True,
    debug_info: List[Tuple[bool, int]] = None,
) -> List[CroppedPage]:
    if debug_info:
        im = page.page_crop.to_image()
        for ints, loc in debug_info:
            if ints:
                if vertical_div:
                    im.draw_hline(loc)
                else:
                    im.draw_vline(loc)
        im.show()

    page_bbox = page.page_crop.bbox

    page_crops = []
    for section in reversed(page_breaks):
        p0, p1 = section

        if vertical_div:
            div_tup = (page_bbox[0], p0, page_bbox[2], p1)
        else:
            div_tup = (p0, page_bbox[1], p1, page_bbox[3])

        crop = page.page_crop.crop(div_tup, relative=False)
        page_crops.append(crop)

    return page_crops


def partition_page(page: Section, debug=False) -> List[CroppedPage]:
    page_breaks, debug_info = pdf_page_scan(
        page.page_crop,
        vertical_scan=page.vertical_seg,
        line_spacing=5.0 if page.vertical_seg else 8.0,  # arbitrary hyperparameters
        debug=debug,
    )
    return section_page(
        page, page_breaks, vertical_div=page.vertical_seg, debug_info=debug_info
    )
