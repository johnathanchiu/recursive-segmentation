from typing import List

from pdfplumber.page import Page
from PIL import Image

from .image import ImageSection, partition_image
from .pdf import Section, PageSection, partition_page


# TODO: add padding argument
def segment_pdf_page(page: Page, debug: bool = False, padding=1) -> List[PageSection]:
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

    ret_parsed_segments = []
    for crop in parsed_segments:
        bbox = crop.bbox
        if padding:
            bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(bbox[2] + padding, page.width),
                min(bbox[3] + padding, page.height),
            )

        ret_parsed_segments.append(
            PageSection(bounding_box=bbox, page_crop=page.crop(bbox, relative=False))
        )

    return ret_parsed_segments


def segment_pdf_image(page_image: Image.Image, padding=1) -> List[ImageSection]:
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

    if padding:
        for pseg in parsed_segments:
            bbox = pseg.bounding_box
            pseg.bounding_box = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(bbox[2] + padding, page_image.width),
                min(bbox[3] + padding, page_image.height),
            )

    return parsed_segments
