import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import random
import shutil

# ==============================
# 1. TỰ CHIA DATA (train -> test)
# ==============================
def is_folder_empty(folder):
    for root, dirs, files in os.walk(folder):
        if len(files) > 0:
            return False
    return True


def split_data(base_dir='data', split=0.8):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # 👉 FIX: kiểm tra có ảnh thật hay không
    if os.path.exists(test_dir) and not is_folder_empty(test_dir):
        print("Test data đã có ảnh -> bỏ qua split")
        return

    print("Test rỗng -> tiến hành chia dữ liệu...")

    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        images = os.listdir(class_path)

        random.shuffle(images)
        split_idx = int(len(images) * split)

        test_images = images[split_idx:]

        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)

            # 👉 dùng copy để an toàn
            shutil.copy(src, dst)

    print("Đã tạo test dataset!")

# ==============================
# 2. DATA AUGMENTATION
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==============================
# 3. CNN MODEL
# ==============================
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================
# 4. MAIN
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_data('data', split=0.8)

    # Load dataset
    train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
    test_dataset = datasets.ImageFolder('data/test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Classes:", train_dataset.classes)

    # Model
    model = CatDogCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # ==============================
    # 5. TRAIN
    # ==============================
    for epoch in range(15):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Acc: {acc:.2f}%")

    # ==============================
    # 6. TEST
    # ==============================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # ==============================
    # 7. SAVE MODEL
    # ==============================
    torch.save(model.state_dict(), "cat_dog_model.pth")
    print("Đã lưu model!")

# ==============================
if __name__ == "__main__":
    main()