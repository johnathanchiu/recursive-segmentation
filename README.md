# Recursive Segmentation Model

The ideas presented in this repository are largely based off the original paper from 1995: _Recursive XY cut using bounding boxes of connected components_ (https://ieeexplore.ieee.org/document/602059). It's a super lightweight segmentation algorithm with no ML components so it also segments extremely fast and can be done in parallel too (more to come on this front).

**_Disclaimer_**: _This is an unbenchmarked segmentation model. It works decently well for documents at first glance and will be extended to general images in the near future. I also need to find a better name for this package._

## Getting Started

This repository is pushed to a PyPI distribution (https://pypi.org/project/xy-segmentation/). Get started by running the following command:

```
pip install xy-segmentation
```

Example usage:

```python
ifile = "examples/images/apple_iphone-13_manual.jpg"
img = Image.open(ifile)

draw = ImageDraw.Draw(img, "RGBA")
for crop in segment_pdf_image(img):
    draw.rectangle(
        crop.bounding_box, outline=(255, 0, 0), width=3, fill=(0, 127, 255, 80)
    )

img.show()
```

## Examples

<p>
<img src="https://github.com/johnathanchiu/recursive-segmentation/blob/main/examples/outputs/apple_output.jpg" alt="Image 1" width="400"/> 
<img src="https://github.com/johnathanchiu/recursive-segmentation/blob/main/examples/outputs/dell_output.jpg" alt="Image 2" width="400"/>
</p>

See `main.py` or `ex.ipynb` for examples on how to draw the images.

Examples from the `pdfs` folder under `examples` were grabbed from [here](https://www.princexml.com/samples/) and `images` folder under `examples` were grabbed from [here](https://github.com/AIM3-RUC/MPMQA).

## Local Setup

```
pip install -r requirements.txt
```

## Additional Information

This algorithm works particularly well with documents that have a lot of diagrams and that are well spaced. It performs poorly on documents that are purely text-based (but there is usually no need to segment documents that are completely text-based just throw it into RAG directly). It could be interesting to detect situations like this and skip the segmentation step entirely for these sorts of pages.

At the moment, I am looking to build out an ML model to determine when to split chunks in the page. The main principle would be to train a seq2seq model that outputs a binary sequence. The sequence input is the slices of the image and the output is a binary sequence where a 1 represents a split in the image and 0 otherwise.

### Limitations

Like any bounding box segmentation algorithm, the main limitation is the shape of the segmentation. Edge cases arise when the input image is not necessarily framed in a grid-shape. Take an example where an image contains "L" shaped objects. This makes it impossible to segment out the "L" shaped object defined by a bounding box. If anyone has any ideas on how to improve this, please feel free to suggest!

## Contributing

Feel free to contribute to this repository through Pull Requests and Issues. Reach out to me if you have any ideas surrounding this that you want to discuss!
