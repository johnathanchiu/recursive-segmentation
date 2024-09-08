from dataclasses import dataclass
from typing import List, Tuple

from pdfplumber.page import CroppedPage, Page
from PIL import Image
import numpy as np

from .scan import image_scan, page_scan


@dataclass
class Section:
    page_crop: CroppedPage
    vertical_seg: bool
    seg_depth: int = 0


@dataclass
class ImageSection:
    bounding_box: Tuple[int, int, int, int]
    page_image: Image.Image
    vertical_seg: bool


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


# TODO: Support image operations for page scan
def partition_page(page: Section, debug=False) -> List[CroppedPage]:
    page_breaks, debug_info = page_scan(
        page.page_crop,
        vertical_scan=page.vertical_seg,
        line_spacing=5.0 if page.vertical_seg else 8.0,  # arbitrary hyperparameters
        debug=debug,
    )
    return section_page(
        page, page_breaks, vertical_div=page.vertical_seg, debug_info=debug_info
    )


def segment_pdf_page(page: Page, debug: bool = False) -> List[CroppedPage]:
    page_queue = [Section(page_crop=page, vertical_seg=True)]

    parsed_segments = []

    count = 0
    while page_queue:
        # Get the next crop in the queue
        curr_crop = page_queue.pop(0)

        # Partition the next crop by the opposite method
        crops = partition_page(curr_crop, debug=debug)

        # if the page cannot be partitioned further than insert it directly into `parsed_segments`
        if len(crops) == 1:
            parsed_segments.append(crops.pop())

        for crop in crops:
            page_queue.append(
                Section(
                    page_crop=crop,
                    vertical_seg=not curr_crop.vertical_seg,
                    seg_depth=curr_crop.seg_depth + 1,
                )
            )

        count += 1

    return parsed_segments


def partition_image(page_section: ImageSection) -> List[Image.Image]:
    page_image = page_section.page_image
    vseg = page_section.vertical_seg

    page_breaks, _ = image_scan(
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


def segment_pdf_image(page_image: Image.Image) -> List[ImageSection]:
    page_queue = [
        ImageSection(
            bounding_box=(0, 0, page_image.width, page_image.height),
            page_image=page_image.convert("L"),
            vertical_seg=True,
        )
    ]

    parsed_segments = []

    count = 0
    while page_queue:
        # Get the next crop in the queue
        curr_crop = page_queue.pop(0)

        # Partition the next crop by the opposite method
        crops = partition_image(curr_crop)

        # if the page cannot be partitioned further than insert it directly into `parsed_segments`
        if len(crops) == 1:
            parsed_segments.append(crops.pop())

        for crop in crops:
            page_queue.append(crop)

        count += 1

    for crop in page_queue:
        parsed_segments.append(crop)

    return parsed_segments
