import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from torchvision.models import ResNet18_Weights

# ---------------------------
# Load Classes
# ---------------------------
data_dir = "dataset"
classes = os.listdir(data_dir)
classes.sort()
print("Loaded classes:", classes)

# ---------------------------
# Load Model
# ---------------------------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/leaf_model.pth"))
model.eval()

# ---------------------------
# Transform Input Image
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# Predict Function
# ---------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    print("\nPrediction:", classes[predicted.item()])


# ---------------------------
# Run Prediction
# ---------------------------
image_path = input("Enter image path: ").strip()

if os.path.exists(image_path):
    predict_image(image_path)
else:
    print("Image not found!")
