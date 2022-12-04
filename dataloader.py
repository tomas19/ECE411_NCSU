# from __future__ import print_function
# from __future__ import division
import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, num_classes):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'Images','*.*'))
        self.label_files = []
        self.num_classes = num_classes
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            label_filename_with_ext = f"{image_filename}_label.jpg"
            self.label_files.append(os.path.join(folder_path, 'Labels', label_filename_with_ext))

        self.transforms = transforms.Compose([transforms.ToTensor()])
 
    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]

            image = Image.open(img_path)
            label = Image.open(label_path)

            # Set pixels with values above the number of classes to the number of clases
            label_np = np.asarray(label)
            label_np = np.clip(label_np, 0, self.num_classes)

            # Apply Transforms
            image = self.transforms(image)
            label = Image.fromarray(label_np)
            label = self.transforms(label).unsqueeze(0)

            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()
            return image, label

    def __len__(self):
        return len(self.img_files)