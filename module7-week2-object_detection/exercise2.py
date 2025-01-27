import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import ResNet18_Weights


# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, annotations_dir, image_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = self.filter_images_with_multiple_objects()

    def filter_images_with_multiple_objects(self):
        valid_image_files = []
        for f in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, f)):
                img_name = f
                annotation_name = os.path.splitext(img_name)[0] + ".xml"
                annotation_path = os.path.join(self.annotations_dir, annotation_name)

                if self.count_objects_in_annotation(annotation_path) == 1:
                    valid_image_files.append(img_name)
        return valid_image_files

    def count_objects_in_annotation(self, annotation_path):
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            count = 0
            for obj in root.findall('object'):
                count += 1
            return count
        except FileNotFoundError:
            return 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Annotation path
        annotation_name = os.path.splitext(img_name)[0] + ".xml"
        annotation_path = os.path.join(self.annotations_dir, annotation_name)

        # Parse annotation
        label, bbox = self.parse_annotation(annotation_path)  # Get both label and bbox

        if self.transform:
            image = self.transform(image)

        return image, label, bbox

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Get image size for normalization
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        label = None
        bbox = None
        for obj in root.findall('object'):
            name = obj.find('name').text
            if label is None:  # Take the first label
                label = name
                # Get bounding box coordinates
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)

                # Normalize bbox coordinates to [0, 1]
                bbox = [
                    xmin / image_width,
                    ymin / image_height,
                    xmax / image_width,
                    ymax / image_height,
                ]

        # Convert label to numerical representation (0 for cat, 1 for dog)
        label_num = 0 if label == 'cat' else 1 if label == 'dog' else -1

        return label_num, torch.tensor(bbox, dtype=torch.float32)
# Model with Two Heads
class TwoHeadedModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TwoHeadedModel, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.num_ftrs = self.base_model.fc.in_features

        # Remove the original fully connected layer
        self.base_model.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Linear(self.num_ftrs, num_classes)

        # Bounding box regression head
        self.regressor = nn.Linear(self.num_ftrs, 4)

    def forward(self, x):
        x = self.base_model(x)
        class_logits = self.classifier(x)
        bbox_coords = torch.sigmoid(self.regressor(x))
        return class_logits, bbox_coords


if __name__ == "__main__":
  data_dir = "data/andrewmvd/dog -and -cat - detection"

  # Data directory
  annotations_dir = os.path.join(data_dir, 'annotations')
  image_dir = os.path.join(data_dir, 'images')

  # Get list of image files and create a dummy dataframe to split the data
  image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
  df = pd.DataFrame({'image_name': image_files})

  # Split data
  train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    
    # Transforms
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # Datasets
  train_dataset = ImageDataset(annotations_dir, image_dir, transform=transform)
  val_dataset = ImageDataset(annotations_dir, image_dir, transform=transform)

  # Filter datasets based on train_df and val_df
  train_dataset.image_files = [f for f in train_dataset.image_files if f in train_df['image_name'].values]
  val_dataset.image_files = [f for f in val_dataset.image_files if f in val_df['image_name'].values]

  # Dataloaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  # Model
  model = TwoHeadedModel()

  # Device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Loss and Optimizer
  criterion_class = nn.CrossEntropyLoss()
  criterion_bbox = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training Loop
  num_epochs = 10
  for epoch in range(num_epochs):
      model.train()
      for batch_idx, (data, targets, bboxes) in enumerate(train_loader):
          data = data.to(device)
          targets = targets.to(device)
          bboxes = bboxes.to(device)

          scores, pred_bboxes = model(data)
          loss_class = criterion_class(scores, targets)
          loss_bbox = criterion_bbox(pred_bboxes, bboxes)
          loss = loss_class + loss_bbox  # Combine losses

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      # Validation
      model.eval()
      with torch.no_grad():
          correct = 0
          total = 0
          total_loss_bbox = 0
          total_samples = 0
          for data, targets, bboxes in val_loader:
              data = data.to(device)
              targets = targets.to(device)
              bboxes = bboxes.to(device)

              scores, pred_bboxes = model(data)
              _, predictions = scores.max(1)
              correct += (predictions == targets).sum()
              total += targets.size(0)

              # Calculate bbox loss for monitoring (optional)
              total_loss_bbox += criterion_bbox(pred_bboxes, bboxes).item() * data.size(0)
              total_samples += data.size(0)

          avg_loss_bbox = total_loss_bbox / total_samples

          print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {float(correct)/float(total)*100:.2f}%, '
                f'Avg. Bbox Loss: {avg_loss_bbox:.4f}')



  
