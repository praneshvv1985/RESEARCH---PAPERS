import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SatelliteImageDataset(Dataset):
    def _init_(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        for label in self.class_names:
            class_path = os.path.join(root_dir, label)
            for img in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img))
                self.labels.append(label)
        self.le = LabelEncoder()
        self.encoded_labels = self.le.fit_transform(self.labels)

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img_path = self.image_paths[idx]
        label = self.encoded_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label_encoder(self):
        return self.le

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = SatelliteImageDataset(root_dir="dataset/all", transform=transform)
le = dataset.get_label_encoder()
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(loader):
            images = images.to(device)
            outputs = resnet(images)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

start_time = time.time()
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)
extraction_time = time.time() - start_time

pca = PCA(n_components=256)
train_features_pca = pca.fit_transform(train_features)
val_features_pca = pca.transform(val_features)

param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}
svm = GridSearchCV(SVC(), param_grid, cv=3, verbose=1, n_jobs=-1)
svm.fit(train_features_pca, train_labels)

preds = svm.predict(val_features_pca)
acc = accuracy_score(val_labels, preds)
report = classification_report(val_labels, preds, target_names=le.classes_)
cm = confusion_matrix(val_labels, preds)

print(f"Feature extraction time: {extraction_time:.2f} seconds")
print(f"Validation Accuracy: {acc}")
print(report)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

joblib.dump(svm, "resnet_svm_model.pkl")
joblib.dump(pca, "resnet_pca.pkl")
