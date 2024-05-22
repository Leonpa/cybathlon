import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        print(f"Initialized dataset with {len(image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

        with open(label_path, 'r') as f:
            label = list(map(float, f.readline().strip().split()))

        if self.transform:
            image = self.transform(image)

        class_label = int(label[0])
        bbox = torch.tensor(label[1:])  # Assuming the label format is [class, x_center, y_center, width, height]

        return image, (class_label, bbox)


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader=None, learning_rate=1e-3,
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
            images, (class_labels, bboxes) = batch
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)
            bboxes = bboxes.to(self.device)
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
                images, (class_labels, bboxes) = batch
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                bboxes = bboxes.to(self.device)
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

    @staticmethod
    def render_image_with_boxes(self, image, true_box, predicted_box):
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())

        def draw_box(box, color):
            x_center, y_center, width, height = box
            x = x_center - width / 2
            y = y_center - height / 2
            rect = patches.Rectangle((x * 128, y * 128), width * 128, height * 128, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Draw true box in green
        draw_box(true_box, 'g')

        # Draw predicted box in red
        draw_box(predicted_box, 'r')

        plt.show()

    def run_inference(self):
        for images, (class_labels, true_boxes) in self.test_loader:
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)
            true_boxes = true_boxes.to(self.device)
            predictions = self.predict(images[0])

            # Debugging: Print shapes and content
            print("Predictions shape:", predictions.shape)
            print("Predictions:", predictions)
            print("Labels:", class_labels[0])
            print("True boxes:", true_boxes[0])

            # Assuming the predictions return bounding box coordinates only
            predicted_boxes = predictions.squeeze().cpu().numpy()

            # Debugging: Ensure we have the correct shapes
            print("Predicted boxes:", predicted_boxes)

            # Render the image with true and predicted boxes
            self.render_image_with_boxes(images[0].cpu(), true_boxes[0].cpu().numpy(), predicted_boxes)
            break  # Render only the first image in the batch for this example
