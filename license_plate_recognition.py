from flask import Flask, request, jsonify
import os
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from fast_plate_ocr import ONNXPlateRecognizer
from util import get_car

# Initialize Flask app
app = Flask(__name__)

# Global model initialization
coco_model = YOLO('yolov8n.pt')  # Path to YOLOv8 COCO model
license_plate_detector = YOLO('license_plate_detector.pt')  # Path to License Plate Detection model
recognizer = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')  # Path to ONNX OCR model
mot_tracker = Sort()  # Vehicle tracking object

# Helper function to draw border
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

# Image processing function
def process_image(file_path, input_filename):
    """
    Process the image to detect vehicles, license plates, and recognize text.

    Args:
        file_path (str): Path to the input image.
        input_filename (str): Original input file name.

    Returns:
        dict: Processed image path and recognized text.
    """
    results = {}

    # Read the input image
    frame = cv2.imread(file_path)
    if frame is None:
        return {'error': 'Image not found or unable to read.'}

    # Detect vehicles using YOLO COCO model
    detections = coco_model(frame)[0]
    detections_ = []
    vehicles = [2, 3, 5, 7]  # Vehicle class IDs for car, bus, truck, motorcycle
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Update SORT tracker
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates using the license plate model
    license_plates = license_plate_detector(frame)[0]
    recognized_texts = []

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign license plate to a car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            result = recognizer.run(license_plate_crop)
            cleaned_result = ''.join(filter(str.isalnum, result)) if result else "No Text"
            recognized_texts.append(cleaned_result)

        # Draw border on license plate
        draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)))

    # Save the processed image with the name process_{input_image_name}.jpg
    processed_image_name = f"process_{input_filename}"
    processed_image_path = os.path.join(tempfile.gettempdir(), processed_image_name)
    cv2.imwrite(processed_image_path, frame)

    return {
        'processed_image_path': processed_image_path,
        'recognized_text': recognized_texts
    }

# Flask route to handle image upload and processing
@app.route('/process-image', methods=['POST'])
def upload_image():
    try:
        # Check if a file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400

        # Save the file temporarily
        input_filename = file.filename
        temp_file_path = os.path.join(tempfile.gettempdir(), input_filename)
        file.save(temp_file_path)

        # Process the image
        result = process_image(temp_file_path, input_filename)

        return jsonify({
            'recognized_text': result['recognized_text'],
            'processed_image_path': result['processed_image_path']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
