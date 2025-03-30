from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import io

app = Flask(__name__)

class HandwrittenDigitCNN(nn.Module):
    def __init__(self):
        super(HandwrittenDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model
model = HandwrittenDigitCNN()
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        return jsonify({"digit": predicted.item(), "confidence": confidence[predicted.item()].item()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
