from dataclasses import dataclass
from typing import List, Tuple

from pdfplumber.page import CroppedPage, Page
from PIL import Image

from .scan import page_scan


@dataclass
class Section:
    page_crop: CroppedPage
    vertical_seg: bool
    seg_depth: int = 0


class ImageSection:
    bounding_box: Tuple[int, int, int, int]


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


# TODO: Support image operations for page scan
def partition_page(page: Section, debug=False) -> List[CroppedPage]:
    page_bbox = page.page_crop.bbox
    if page.vertical_seg:
        page_dim = (page_bbox[1], page_bbox[3])
        line_spacing = 5.0  # arbitrary hyperparameters
    else:
        page_dim = (page_bbox[0], page_bbox[2])
        line_spacing = 8.0  # arbitrary hyperparameters

    page_breaks, debug_info = page_scan(
        page.page_crop.objects,
        page_dim,
        vertical_scan=page.vertical_seg,
        line_spacing=line_spacing,
        debug=debug,
    )
    return section_page(
        page, page_breaks, vertical_div=page.vertical_seg, debug_info=debug_info
    )


def partition_image(page: Image, vertical_scan=True):
    raise NotImplementedError()


def segment_pdf_page(page: Page, debug=False) -> List[CroppedPage]:
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
