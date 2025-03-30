import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image, ImageOps
import psycopg2
from streamlit_drawable_canvas import st_canvas

class HandwrittenDigitCNN(nn.Module):
    def __init__(self):
        super(HandwrittenDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_data
def load_trained_model():
    model = HandwrittenDigitCNN()
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    return transform(image).unsqueeze(0)

def predict_digit(image):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return predicted.item(), confidence[predicted.item()].item()

DB_CONFIG = {
    "dbname": "predictions_db",
    "user": "postgres",
    "password": "agunpass",
    "host": "127.0.0.1",
    "port": "5432"
}

def log_prediction(predicted_digit, true_label, confidence):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (predicted_digit, true_label, confidence) VALUES (%s, %s, %s)",
                       (predicted_digit, true_label, confidence))
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def fetch_predictions():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id, predicted_digit, true_label, confidence, timestamp FROM predictions ORDER BY timestamp DESC")
        records = cursor.fetchall()
        return records
    except Exception as e:
        st.error(f"Database error: {e}")
        return []
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

st.title("Handwritten Digit Recognition")

uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([3, 1])

with col1:
    canvas_result = st_canvas(
        width=280, height=280, drawing_mode="freedraw", stroke_width=10, stroke_color="black", background_color="white"
    )

with col2:
    true_label = st.number_input("True Label (Optional)", min_value=0, max_value=9, step=1)

if st.button("Submit Prediction"):
    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
    elif canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype('uint8'))
    
    if image is not None:
        predicted_digit, confidence = predict_digit(image)
        st.write(f"Predicted Digit: **{predicted_digit}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        log_prediction(predicted_digit, true_label, confidence)
        st.success(f"Prediction logged successfully!")
    else:
        st.warning("Please upload an image or draw a digit before submitting.")

if st.button("Show Previous Predictions"):
    records = fetch_predictions()
    
    if records:
        st.write("Prediction History")
        for record in records:
            st.write(
                f"ID: {record[0]} | Predicted: {record[1]} | True: {record[2]} | Confidence: {record[3]:.2f}% | Time: {record[4] if record[4] else 'N/A'}"
            )
    else:
        st.info("No predictions found in the database.")
