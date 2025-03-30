# Handwritten Digit Recognition

This is a Streamlit web app that recognizes handwritten digits using a Convolutional Neural Network (CNN). 
The app allows users to upload an image or draw a digit on an interactive canvas, and it predicts the digit with a confidence score.

http://95.216.222.81:8501/

🖥️ Features
✅ Upload an image of a handwritten digit for recognition.
✅ Draw a digit on the canvas for instant prediction.
✅ View prediction history stored in a PostgreSQL database.

🛠️ Installation & Usage

* Clone the repository:
git clone https://github.com/augun/handwritten-digit-streamlit.git

* Navigate into the project directory:
cd handwritten-digit-streamlit

* Set up a virtual environment:
python3 -m venv venv

* Activate the virtual environment:
source venv/bin/activate

* Install dependencies:
pip install -r requirements.txt

* Run the app:
streamlit run app.py

* Open the app in your browser:
http://localhost:8501/

🎯 Model Information

- The model is a CNN trained on the MNIST dataset.
- It consists of convolutional, pooling, and fully connected layers.
- The trained model weights are stored in trained_model.pth.



