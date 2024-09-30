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
