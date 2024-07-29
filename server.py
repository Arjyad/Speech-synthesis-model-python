import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import torch
import numpy as np
from model import Generator

app = Flask(__name__)
CORS(app)

CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)  # Add this line for detailed logging

# Define directories
HOME_DIR = os.path.expanduser("~")
UPLOAD_FOLDER = os.path.join(HOME_DIR, 'uploaded_files')
RESULTS_FOLDER = os.path.join(HOME_DIR, 'results')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(input_shape=(80, 64)).to(device)

@app.route('/convert', methods=['POST'])
def convert():
    
    if 'source_file' not in request.files or 'target_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    source_file = request.files['source_file']
    target_file = request.files['target_file']

    if source_file:
        print('source file present')
    if target_file:
        print('target file present')

    if source_file.filename == '' or target_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        source_path = os.path.join(UPLOAD_FOLDER, source_file.filename)
        target_path = os.path.join(UPLOAD_FOLDER, target_file.filename)
        source_file.save(source_path)
        target_file.save(target_path)
    except Exception as e:
        app.logger.error(f"File save error: {str(e)}")
        return jsonify({'error': 'Failed to save files'}), 500

    try:
        source_data = np.load(source_path)  # Adjust based on data format
        target_data = np.load(target_path)  # Adjust based on data format

        source_tensor = torch.from_numpy(source_data).float().to(device)
        target_tensor = torch.from_numpy(target_data).float().to(device)
        mask = torch.ones_like(source_tensor).to(device)

        output = generator(source_tensor, mask)
        output_path = os.path.join(RESULTS_FOLDER, "output.npy")
        output_np = output.cpu().detach().numpy()
        np.save(output_path, output_np)
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': 'Failed to process files'}), 500

    if os.path.exists(output_path):
        return jsonify({'outputFileUrl': f'http://localhost:5000/results/output.npy'})
    else:
        return jsonify({'error': 'Output file not found'}), 404

@app.route('/results/<path:filename>')
def download_file(filename):
    try:
        return send_from_directory(RESULTS_FOLDER, filename)
    except Exception as e:
        app.logger.error(f"File serve error: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
