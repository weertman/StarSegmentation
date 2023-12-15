from dataloader import COCO_SegmentationDataset
from visualizations import COCO_SegmentationDataset_Visualizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_batch(batch, dataset):
    images, targets = batch
    images = images.numpy()
    batch_size = len(images)

    # Assuming a grid layout for visualization
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i in range(grid_size * grid_size):
        ax = axs[i // grid_size, i % grid_size]
        if i < batch_size:
            # Convert image from PyTorch tensor to numpy array
            img = np.transpose(images[i], (1, 2, 0))

            # Normalize the image data to [0, 1] if it's not already
            if img.max() > 1.0:
                img = img / img.max()

            # Display image
            ax.imshow(img)

            # Get annotations for this image
            boxes = targets['boxes'][i].numpy()
            masks = targets['masks'][i].numpy()
            labels = targets['labels'][i].numpy()

            colors = {}

            # Draw bounding boxes and masks
            for box, mask, label_id in zip(boxes, masks, labels):
                label_name = dataset.get_category_name(label_id)
                if label_id not in colors:
                    colors[label_id] = (random.random(), random.random(), random.random())

                # Check and process the mask
                if mask.max() > 1:
                    mask = mask / 255.0

                colored_mask = np.zeros_like(img)
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask == 1, colors[label_id][c], 0)

                # Draw rectangle for bounding box
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                         linewidth=2, edgecolor=colors[label_id], facecolor='none')
                ax.add_patch(rect)
                ax.text(box[0], box[1], label_name, color='white', fontsize=12,
                        bbox=dict(facecolor=colors[label_id], edgecolor='none', boxstyle='round,pad=0.5'))

                # Overlay mask
                ax.imshow(colored_mask, alpha=0.1)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = {key: [item[1][key] for item in batch] for key in batch[0][1].keys()}
    return images, targets

def main():
    dataset_root = '../../datasets/StarSeg.v22i.coco-segmentation'

    # For training data
    train_dataset = COCO_SegmentationDataset(root=dataset_root, split='train')
    # For validation data
    valid_dataset = COCO_SegmentationDataset(root=dataset_root, split='valid')
    # For test data
    test_dataset = COCO_SegmentationDataset(root=dataset_root, split='test')

    n_epochs = 10  # Number of epochs
    batch_size = 4  # Adjust this based on your requirement and GPU memory
    shuffle = True  # Shuffle the data every epoch

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=custom_collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=4)

    visualize = False
    if visualize:
        print("Visualizing dataset samples...")
        COCO_SegmentationDataset_Visualizer(train_dataset, min_annotations=2, split='train')
        COCO_SegmentationDataset_Visualizer(valid_dataset, min_annotations=2, split='valid')
        COCO_SegmentationDataset_Visualizer(test_dataset, min_annotations=2, split='test')

    # Placeholder for best model (to be implemented later)
    best_model = None

    # Training loop
    pbar = tqdm(total=n_epochs, desc="Training progress", unit="epoch", position=0, leave=True)
    for epoch in range(n_epochs):
        # Iterate over training data
        bbar = tqdm(total=len(train_loader), desc="Epoch {}/{}".format(epoch + 1, n_epochs), unit="batch", position=1, leave=True)
        for i, batch in enumerate(train_loader):
            # Training step (to be implemented)
            if i == 0:
                visualize_batch(batch)
            bbar.update(1)
            pass
        bbar.close()

        # Evaluate on validation data after each epoch
        vbar = tqdm(total=len(valid_loader), desc="Epoch {}/{}".format(epoch + 1, n_epochs), unit="batch", position=1, leave=True)
        for i, batch in enumerate(valid_loader):
            # Validation step (to be implemented)
            if i == 0:
                visualize_batch(batch)
            vbar.update(1)
            pass
        vbar.close()

        # Save best model (to be implemented)
        pbar.update(1)
    pbar.close()

    # Evaluate the best model on the test data
    for batch in test_loader:
        # Test step (to be implemented)
        if i == 0:
            visualize_batch(batch)
        pass

    print("Training completed.")

if __name__ == '__main__':
    main()

