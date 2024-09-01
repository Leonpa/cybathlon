import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json


class Model(nn.Module):
    def __init__(self, num_channels=3, num_classes=4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, coco_json, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load COCO JSON
        with open(coco_json, 'r') as f:
            self.coco_data = json.load(f)

        # Create a dictionary for images and annotations
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            if ann['image_id'] not in self.annotations:
                self.annotations[ann['image_id']] = []
            self.annotations[ann['image_id']].append(ann)

        self.image_ids = list(self.images.keys())
        print(f"Initialized dataset with {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        labels = []
        if image_id in self.annotations:
            for ann in self.annotations[image_id]:
                category_id = ann['category_id']
                bbox = torch.tensor(ann['bbox'])  # x, y, width, height
                labels.append((category_id, bbox))

        # Apply transformation to the image if any
        if self.transform:
            image = self.transform(image)

        # Return image and list of tuples (category_id, bbox)
        return image, labels


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader=None, learning_rate=0.001,
                 device="cuda" if torch.cuda.is_available() else "cpu", lr_step_size=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.device = device
        self.step_size = lr_step_size

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=0.1)
        self.loss_history = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            images, labels = batch
            images = images.to(self.device)
            class_labels = torch.tensor([lbl[0] for lbl in labels]).to(self.device)
            bboxes = torch.stack([lbl[1] for lbl in labels]).to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, class_labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.loss_history.append(epoch_loss)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch
                images = images.to(self.device)
                class_labels = torch.tensor([lbl[0] for lbl in labels]).to(self.device)
                bboxes = torch.stack([lbl[1] for lbl in labels]).to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, class_labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()
        epoch_loss = running_loss / len(self.val_loader.dataset)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate_epoch()
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            self.scheduler.step()
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_loss.png')
        plt.close()


class Inference:
    def __init__(self, model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image.unsqueeze(0))
        return output

    def render_image_with_boxes(self, image, true_box, predicted_box):
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())

        def draw_box(box, color):
            x, y, width, height = box
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Draw true box in green
        draw_box(true_box, 'g')

        # Draw predicted box in red
        draw_box(predicted_box, 'r')

        plt.show()

    def run_inference(self, sample_indices=None, num_samples=1):
        if sample_indices:
            samples = [self.test_loader.dataset[i] for i in sample_indices]
        else:
            samples = [self.test_loader.dataset[i] for i in range(num_samples)]

        for image, labels in samples:
            image = image.to(self.device)
            for class_label, true_box in labels:
                predictions = self.predict(image)

                # Assuming the predictions return bounding box coordinates only
                predicted_box = predictions.squeeze().cpu().numpy()

                # Render the image with true and predicted boxes
                self.render_image_with_boxes(image.cpu(), true_box.cpu().numpy(), predicted_box)
