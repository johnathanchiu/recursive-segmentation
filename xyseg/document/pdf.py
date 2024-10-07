from dataclasses import dataclass
from typing import List, Tuple

from pdfplumber.page import CroppedPage
import numpy as np

from .div import div_intersections


@dataclass
class Section:
    page_crop: CroppedPage
    vertical_seg: bool
    seg_depth: int = 0


@dataclass
class PageSection:
    bounding_box: Tuple[int, int, int, int]
    page_crop: CroppedPage


def check_object_intersections(page_objs, scan_line, p0, p1):
    is_crossed = False
    for obj_type in page_objs:
        # We only check objects that fall into these categories
        if obj_type not in {"line", "curve", "rect", "char", "image"}:
            continue
        for obj in page_objs[obj_type]:
            if obj[p0] < scan_line < obj[p1]:
                is_crossed = True
                break
        if is_crossed:
            break
    return is_crossed


def pdf_page_scan(page: CroppedPage, line_spacing=5.0, vertical_scan=True, debug=False):
    # vertical scan implies the lines are going across the page dropped from top to bottom
    page_bbox = page.bbox
    page_objs = page.objects
    if vertical_scan:
        p0, p1 = "top", "bottom"
        page_dim = (page_bbox[1], page_bbox[3])
        line_spacing = 5.0  # arbitrary hyperparameters
    else:
        p0, p1 = "x0", "x1"
        page_dim = (page_bbox[0], page_bbox[2])
        line_spacing = 8.0  # arbitrary hyperparameters

    scan_intersects = []
    scan_lines = list(np.arange(*page_dim, line_spacing))
    for scan_line in scan_lines:
        is_crossed = check_object_intersections(page_objs, scan_line, *(p0, p1))
        scan_intersects.append(is_crossed)

    debug_info = None
    if debug:
        debug_info = zip(scan_intersects, scan_lines)

    return div_intersections(scan_intersects, scan_lines), debug_info


def section_page(
    page: Section, page_breaks, vertical_div=True, debug_info=None
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
