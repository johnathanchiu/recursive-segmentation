from typing import List


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
