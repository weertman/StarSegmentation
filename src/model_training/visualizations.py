import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class COCO_SegmentationDataset_Visualizer:
    def __init__ (self, dataset, split, min_annotations, stop_after=100):
        self.dataset = dataset
        self.min_annotations = min_annotations
        self.split = split
        self.stop_after = stop_after
        self.idx = self.find_image_with_multiple_annotations()

        if self.idx != -1:
            #print(f"Visualizing {self.split} image at index {self.idx}.")
            self.visualize_sample()
        else:
            print("No image with the desired number of annotations was found.")

    def visualize_sample(self):
        img, target = self.dataset[self.idx]
        # Convert the tensor image to numpy array and change dimension order
        img = img.permute(1, 2, 0).numpy()

        img = np.array(img)  # Convert PIL image to numpy array

        # Normalize the image data to [0, 1] if it's not already
        if img.max() > 1.0:
            img = img / img.max()
            print("Normalizing image data to [0, 1]; max value: {img.max()}.")

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        colors = {}

        for i in range(len(target['masks'])):
            mask = target['masks'][i].numpy()
            rect = target['boxes'][i].numpy()
            label_id = target['labels'][i].item()

            label_name = self.dataset.get_category_name(label_id)

            if label_id not in colors:
                colors[label_id] = (random.random(), random.random(), random.random())

            # Check and process the mask
            if mask.max() > 1:
                mask = mask / 255.0  # Normalize mask if it's not in binary format
                #print(f"Normalizing mask data to [0, 1]; max value: {mask.max()}.")

            colored_mask = np.zeros_like(img)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask == 1, colors[label_id][c], 0)

            bbox_rect = patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], linewidth=2, edgecolor=colors[label_id], facecolor='none')
            ax.add_patch(bbox_rect)

            ax.text(rect[0], rect[1], label_name, color='white', fontsize=12,
                    bbox=dict(facecolor=colors[label_id], edgecolor='none', boxstyle='round,pad=0.5'))

            ax.imshow(colored_mask, alpha=0.1)

        ax.set_title(f"Image {self.idx} from the {self.split} split")

        plt.show()
        plt.close()

    def find_image_with_multiple_annotations(self):
        """
        Finds the index of the first image in the dataset with at least `min_annotations` annotations.

        Args:
        dataset (CustomDataset): The dataset to search.
        min_annotations (int): Minimum number of annotations required.

        Returns:
        int: The index of the image, or -1 if no such image is found.
        """
        idxs = []
        for idx in range(len(self.dataset)):
            img, target = self.dataset[idx]
            if len(target['masks']) >= self.min_annotations:
                idxs.append(idx)
            if len(idxs) >= self.stop_after:
                break
        idx = random.choice(idxs) if len(idxs) > 0 else -1
        if idx == -1:
            print(f"No image with at least {self.min_annotations} annotations was found.")
            return idx
        else:
            print(f"Found {len(idxs)} images with at least {self.min_annotations} annotations.")
            return idx
