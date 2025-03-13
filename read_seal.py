import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Path to model & images
model_path = 'classification/classification_model.pth'
images_folder = 'images'

# Load Resnet18 model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 25)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Inference
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        return predicted.item()

list_chars = ['Ι', 'Θ', 'Κ', 'Λ', 'C moon-shaped sigma', 'V = Y',
              'Α', 'Ε', 'Η', 'Τ', 'Ρ = rho', 'Ο', 'bg', 'Ν', 'ω',
              'Γ', 'Π', 'R = βῆτα', 'Φ', 'ligature OU', 'Croisette',
              'Μ', 'Δ', 'Χ', 'S = καί']

# Maximum number of images to predict (if all, use len(os.listdir(images_folder)) )
max_image = 100

# Prediction
for k, image_name in enumerate(os.listdir(images_folder)):
    image_path = os.path.join(images_folder, image_name)
    prediction = predict_image(image_path)
    print(f'Image: {image_name}, Prédiction: {list_chars[prediction]}')
    if k == max_image:
        break
