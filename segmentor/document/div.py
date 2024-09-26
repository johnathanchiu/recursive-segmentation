from typing import List


def div_intersections(
    intersections: List[bool], scan_lines: List[float], padding=1
) -> List[tuple]:
    """
    Splits consecutive True values in `intersections` into groups and maps
    those groups to start and end positions from `scan_lines`.
    """
    sections = []
    crop_dimensions = []

    # Collect consecutive True sections in intersections
    start = None
    for i, intersects in enumerate(intersections):
        if intersects and start is None:
            start = i
        elif not intersects and start is not None:
            sections.append((start, i - 1))
            start = None

    # Handle the case where the last group extends to the end of the list
    if start is not None:
        sections.append((start, len(intersections) - 1))

    # Convert sections to crop dimensions using scan_lines
    for start, end in sections:
        p0 = scan_lines[max(0, start - padding)]
        p1 = scan_lines[min(end + padding, len(scan_lines) - 1)]
        crop_dimensions.append((int(p0), int(p1)))

    return crop_dimensions
