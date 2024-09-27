from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Define a transform to preprocess the images
transform = transforms.Compose(
    [
        transforms.RandomCrop(128),  # Randomly crop images to 128x128
        transforms.ToTensor(),  # Convert images to tensor
    ]
)


# Create a custom dataset class
class DocumentSegmentationDataset(Dataset):
    def __init__(self, root_dir, bbox_data, transform=None):
        self.root_dir = root_dir
        self.bbox_data = bbox_data  # Dictionary mapping image names to bounding boxes
        self.transform = transform
        self.image_files = os.listdir(root_dir)  # List all image files in the directory

    def __len__(self):
        return len(self.image_files)  # Return the total number of images

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.image_files[idx]
        )  # Get image file path
        image = Image.open(img_name)  # Open the image

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        # Assuming the model expects strips of the image
        strips, labels = self.create_strips_and_labels(
            image, img_name
        )  # Create strips and labels

        return strips, labels  # Return the strips and their corresponding labels

    def create_strips_and_labels(self, image, img_name):
        # Convert image to numpy array for processing
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        strip_height = 32  # Define the height of each strip
        strips = []
        labels = []

        # Get bounding boxes for the current image
        bboxes = self.bbox_data.get(
            os.path.basename(img_name), []
        )  # Get bounding boxes for the image

        # Create strips from the image
        for y in range(0, height, strip_height):
            strip = image_array[y : y + strip_height, :]  # Get a strip
            strips.append(strip)

            # Check if any bounding box intersects with the current strip
            label = self.check_intersection(bboxes, y, strip_height)
            labels.append(label)

        return (
            strips,
            labels,
        )  # Return the list of strips and their corresponding labels

    def check_intersection(self, bboxes, y, strip_height):
        # Check if any bounding box intersects with the strip
        for bbox in bboxes:
            # bbox format: [x_min, y_min, x_max, y_max]
            if (
                bbox[1] < y + strip_height and bbox[3] > y
            ):  # Check for vertical intersection
                return 1  # There is a page break
        return 0  # No page break
