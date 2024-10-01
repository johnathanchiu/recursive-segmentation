from io import BytesIO
import json
import random

import numpy as np
import requests
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from PIL import Image

from model.seg.imutils import vstrips_process, hstrips_process


class BoundingBoxDataset(Dataset):
    def __init__(self, json_file: str):
        with open(json_file, "r") as f:
            self.data_json = json.load(f)

    def __len__(self):
        return len(self.data_json)  # Return the total number of images

    def __getitem__(self, idx):
        # TODO: This should actually be multiple bboxes, not just a single one
        bbox = self.data_json["annotations"][idx]["bbox"]
        response = requests.get(self.data_json["images"][idx]["coco_url"])
        img = Image.open(BytesIO(response.content))

        crop_dims = RandomCrop.get_params(img, (256, 256))
        crop_dims = [
            crop_dims[0],
            crop_dims[1],
            crop_dims[0] + crop_dims[2],
            crop_dims[1] + crop_dims[3],
        ]
        img = img.crop(crop_dims)

        shifted_bbox = [
            bbox[0] - crop_dims[0],
            bbox[1] - crop_dims[1],
            bbox[2],
            bbox[3],
        ]
        is_vert = random.choice([True, False])
        strips, labels = self.create_strips_and_labels(img, shifted_bbox, is_vert)
        return is_vert, strips, labels

    def create_strips_and_labels(self, img, bbox, is_vert):
        if is_vert:
            return hstrips_process(img, bbox)
        return vstrips_process(img, bbox)


if __name__ == "__main__":
    dataset = BoundingBoxDataset("instances_minitrain2017.json")
    is_vertical, strips, labels = dataset[0]

    bound_strips = []
    for strip, label in zip(strips, labels):
        if label:
            bound_strips.append(np.zeros_like(strip))
        else:
            bound_strips.append(strip)

    if is_vertical:
        img = Image.fromarray(np.vstack(bound_strips))
    else:
        img = Image.fromarray(np.hstack(bound_strips))
    img.show()
