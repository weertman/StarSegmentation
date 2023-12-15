import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import cv2
import numpy as np

class COCO_SegmentationDataset(Dataset):
    def __init__(self, root, split='train', image_size=(500, 500), transform=None):
        self.root = os.path.join(root, split)
        self.image_size = image_size
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
            ])
        else:
            self.transform = transform
        # Load the JSON file
        annotations_file = os.path.join(self.root, "_annotations.coco.json")
        with open(annotations_file) as f:
            self.data = json.load(f)

        self.images = self.data['images']
        self.annotations = self.data['annotations']
        # Create a mapping from image id to annotations
        self.img_to_anns = {img['id']: [] for img in self.images}
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Apply transformation
        img = self.transform(img)

        # Get corresponding annotations
        anns = self.img_to_anns[img_info['id']]

        # Initialize masks, labels, boxes, and area
        num_objs = len(anns)
        masks = torch.zeros((num_objs,
                             img_info['height'],
                             img_info['width']),
                            dtype=torch.uint8)
        boxes = []
        labels = []
        area = []

        for i, ann in enumerate(anns):
            # Initialize masks tensor with the resized image dimensions (500x500)
            masks = torch.zeros((num_objs, self.image_size[0], self.image_size[1]), dtype=torch.uint8)

            # Convert the COCO polygon segmentation to a binary mask
            original_mask = self.decode_segmentation(ann['segmentation'], img_info['height'], img_info['width'])

            # Resize the mask to match the transformed image size
            resized_mask = cv2.resize(original_mask, (self.image_size[1], self.image_size[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # Add the resized mask to the masks tensor
            masks[i] = torch.as_tensor(resized_mask, dtype=torch.uint8)

            # Extract bounding box
            bbox = ann['bbox']

            # Calculate scale factors for height and width
            scale_factor_x = self.image_size[1] / img_info['width']
            scale_factor_y = self.image_size[0] / img_info['height']

            # Extract and rescale bounding box
            # Extract original bounding box
            bbox = ann['bbox']

            # Rescale bounding box coordinates
            rescaled_bbox = [
                bbox[0] * scale_factor_x,
                bbox[1] * scale_factor_y,
                (bbox[0] + bbox[2]) * scale_factor_x,  # x_max
                (bbox[1] + bbox[3]) * scale_factor_y  # y_max
            ]
            boxes.append(rescaled_bbox)

            # Get label and area
            labels.append(ann['category_id'])
            area.append(ann['area'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)

        # Define the target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_info['id']])
        target['area'] = area
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)

        return img, target

    def get_category_name(self, category_id):
        for category in self.data['categories']:
            if category['id'] == category_id:
                return category['name']
        return "Unknown"

    def decode_segmentation(self, segmentation, height, width):
        """
        Converts COCO polygon annotations to a binary mask.

        Args:
        segmentation (list): List of polygons, each represented by a list of points.
        height (int): Height of the image.
        width (int): Width of the image.

        Returns:
        np.ndarray: Binary mask of the object.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            # Flatten the list of points and convert to integers
            np_polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [np_polygon], 255)  # 255 for object area, 0 for background
        return mask

    def custom_collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        images = torch.stack(images, dim=0)

        batched_targets = {}
        for key in targets[0].keys():
            if key == 'masks':
                # Handle masks separately since they can have varying shapes
                batched_targets[key] = [target[key] for target in targets]
            else:
                # Stack other elements that have consistent shapes
                batched_targets[key] = torch.stack([target[key] for target in targets], dim=0)

        return images, batched_targets

