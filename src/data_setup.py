import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DocumentDataset(Dataset):
    def __init__(self,
                 image_dir,
                 annotations_file,
                 transform = None,
                 target_size = (512, 512)):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size

        # load the annotations file
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # get annotation
        annotation = self.annotations[idx]

        # load image
        image_path = os.path.join(self.image_dir, annotation["image"])
        image = Image.open(image_path).convert('RGB')

        # transform
        if self.transform:
            image = self.transform(image)

        # get dimensions
        orig_width, orig_height = annotation["width"], annotation["height"]
        original_size_tensor = torch.tensor([orig_width, orig_height], dtype=torch.float32)

        # get corners
        corners = np.array(annotation["corners"], dtype=np.float32)

        # normalize corners into [0, 1] (x and y axis)
        corners[:, 0] /= orig_width
        corners[:, 1] /= orig_height

        # corners into tensors
        corners_tensor = torch.tensor(corners, dtype=torch.float32)

        return {
            "image": image,
            "corners": corners_tensor,
            "image_name": annotation["image"],
            "original_size": original_size_tensor
        }

def get_transforms(train=True,
                   target_size=(512, 512)):
    if train:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def create_dataloaders(data_dir: str = 'data',
                       batch_size: int = 32,
                       num_workers: int = os.cpu_count(),
                       train_split: float = 0.8,
                       target_size=(512, 512)):
    image_dir = os.path.join(data_dir, 'images')
    annotations_file = os.path.join(data_dir, 'annotations', 'annotations.json')

    # load all annotations
    with open(annotations_file, 'r') as f:
        all_annotations = json.load(f)

    # split the data
    split_idx = int(len(all_annotations) * train_split)
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]

    # create temp annotation files
    train_ann_file = os.path.join(data_dir, 'annotations', 'train_annotations.json')
    val_ann_file = os.path.join(data_dir, 'annotations', 'val_annotations.json')

    with open(train_ann_file, 'w') as f:
        json.dump(train_annotations, f)
    
    with open(val_ann_file, 'w') as f:
        json.dump(val_annotations, f)
    
    # create datasets
    train_dataset = DocumentDataset(
        image_dir=image_dir,
        annotations_file=train_ann_file,
        transform=get_transforms(train=True, target_size=target_size),
        target_size=target_size
    )

    val_dataset = DocumentDataset(
        image_dir=image_dir,
        annotations_file=val_ann_file,
        transform=get_transforms(train=False, target_size=target_size),
        target_size=target_size
    )

    # create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Train split: {train_split}")
    print(f"Target size: {target_size}")

    # remove temp annotation files
    # os.remove(train_ann_file)
    # os.remove(val_ann_file)

    return train_dataloader, val_dataloader