import streamlit as st
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import threading

# Initialize Flask app
flask_app = Flask(__name__)
CORS(flask_app)  # Enable CORS to allow communication from frontend

# Load the TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Function to extract text from individual word images using TrOCR
def extract_text(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return predicted_text

# Define a POST endpoint to receive multiple image blobs
@flask_app.route('/extract', methods=['POST'])
def api_extract_text():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []
    for file in request.files.getlist('files'):  # Handle multiple images
        img_data = file.read()
        extracted_text = extract_text(img_data)
        results.append(extracted_text)

    return jsonify({'texts': results})  # Return the list of extracted texts

# Run the Flask server in a separate thread
def run_flask():
    flask_app.run(port=5001)

thread = threading.Thread(target=run_flask)
thread.start()

# Streamlit UI (optional for testing directly with the backend)
st.title("Text Extraction API with TrOCR")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_data = uploaded_file.read()
    
    extracted_text = extract_text(image_data)
    st.write("Extracted Text:")
    st.write(extracted_text)
