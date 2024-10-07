# Recursive Segmentation Model

The ideas presented in this repository are largely based off the original paper from 1995: _Recursive XY cut using bounding boxes of connected components_ (https://ieeexplore.ieee.org/document/602059).

**_Disclaimer_**: _This is an unbenchmarked segmentation model. It works decently well for documents at first glance and will be extended to general images in the near future. I also need to find a better name for this package._

## Getting Started

This repository is pushed to a PyPI distribution.

## Examples

See `main.py` for examples on how to draw the images.

## Local Setup

```
pip install -r requirements.txt
```

## Additional Information

This algorithm works particularly well with documents that have a lot of diagrams and that are well spaced. It performs poorly on documents that are purely text-based perform poorly.

At the moment, I am looking to build out an ML model to determine when to split chunks in the page. The main principle would be to train a seq2seq model that outputs a binary sequence. The sequence input is the slices of the image and the output is a binary sequence where a 1 represents a split in the image and 0 otherwise.

Like any bounding box segmentation algorithm, the main limitation is the shape of the segmentation. Edge cases arise when the input image is not necessarily framed in a grid-shape. Take an example where an image contains "L" shaped objects. This makes it impossible to segment out the "L" shaped object defined by a bounding box. If anyone has any ideas on how to improve this, please feel free to suggest!

## Contributing

Feel free to contribute to this repository through Pull Requests and Issues. Reach out to me if you have any ideas surrounding this that you want to discuss!
