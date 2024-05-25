from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import cv2


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLOv8 model
model_xray = YOLO('best.pt')
model_rgb = YOLO('best1.pt')


def classify_image(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Calculate mean of R, G, B channels
    mean_r = np.mean(image_np[:, :, 0])
    mean_g = np.mean(image_np[:, :, 1])
    mean_b = np.mean(image_np[:, :, 2])

    # Determine class based on mean values
    if mean_r == mean_g == mean_b:
        return "X-ray"
    else:
        return "Camera"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')




# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return "No file part", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file", 400
#
#     # Open the image file
#     img = Image.open(file.stream).convert("L")  # Convert to grayscale
#
#     # Perform inference with a confidence threshold of 0.2
#     results = model(img, conf=0.2)
#
#     # Plot the results on the image
#     results_plotted = results[0].plot()  # Plotting directly on the image
#
#     # Convert plotted results back to PIL Image
#     result_img = Image.fromarray(results_plotted)
#
#     # Save the result image
#     result_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     result_img.save(result_path)
#
#     return render_template('detect.html', result_image=file.filename)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Open the image file
    img = Image.open(file.stream)

    # Classify the image
    image_class = classify_image(img)

    if image_class == "X-ray":
        model = model_xray
        img = img.convert("L")
    else:
        model = model_rgb

    # if image_class == "X-ray":
    #     model = model_xray
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convert X-ray to grayscale
    # else:
    #     model = model_rgb

        # img = img.convert("RGB")  # Ensure the image is in RGB format

    # Perform inference with a confidence threshold of 0.2
    results = model(img, conf=0.2)

    # Plot the results on the image
    results_plotted = results[0].plot()  # Plotting directly on the image

    # Convert plotted results back to PIL Image

    result_img = Image.fromarray(results_plotted)

    # if image_class == "X-ray":
    #     result_img = Image.fromarray(results_plotted, 'L')
    #
    # else:
    #     result_img = Image.fromarray(results_plotted, 'RGB')
    if image_class =="Camera":
        result_img = Image.fromarray(cv2.cvtColor(results_plotted, cv2.COLOR_BGR2RGB).astype(np.uint8))

    # Save the result image
    result_filename = file.filename.split('.')[0] + '_result.jpg'
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    result_img.save(result_path)

    # Pass the filenames to detect.html
    return render_template('detect.html', result_image=result_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
