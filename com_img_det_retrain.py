import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import math
from PIL import Image
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import shutil


com_img_det_retrain = Flask(__name__)
CORS(com_img_det_retrain)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

com_img_det_retrain.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
com_img_det_retrain.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

ORIGINAL_FOLDER = os.path.join(STATIC_FOLDER, "original")
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

RETRAIN_FOLDER = "retraining_data"
RETRAIN_IMAGE_FOLDER = os.path.join(RETRAIN_FOLDER, "images")

os.makedirs(RETRAIN_IMAGE_FOLDER, exist_ok=True)

RETRAIN_META = os.path.join(RETRAIN_FOLDER, "metadata.csv")
# -------------------------
# LOAD SINGLE COMBINED MODEL
# -------------------------
try:
    combined_model_path = "runs/pole_fire_wire/best_pole_fire_wire.pt"
    model = YOLO(combined_model_path)
    print("Unified YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading combined model: {e}")
    model = None

# Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import shutil

def clear_all_image_folders():
    folders = [
        UPLOAD_FOLDER,
        STATIC_FOLDER,
        ORIGINAL_FOLDER
    ]

    # Delete files inside each folder
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    # IMPORTANT: Recreate required folders if removed
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
# POLE PROPERTY CALCULATIONS

def calculate_pole_properties(cropped_pole_image):
    if cropped_pole_image is None or cropped_pole_image.shape[0] == 0 or cropped_pole_image.shape[1] == 0:
        return 0.0
    gray = cv2.cvtColor(cropped_pole_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    tilt_angle = 0.0
    if lines is not None:
        longest_line = None
        max_line_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            current_line_length = math.hypot(x2 - x1, y2 - y1)
            if current_line_length > max_line_length:
                max_line_length = current_line_length
                longest_line = line[0]
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            dx = x2 - x1
            dy = y2 - y1
            angle_rad = math.atan2(dx, dy)
            angle_deg = abs(math.degrees(angle_rad))
            tilt_angle = 90 - angle_deg if angle_deg <= 90 else angle_deg - 90
    return tilt_angle


def estimate_real_height(pixel_height, ref_pixel_height, ref_real_height_m):
    if ref_pixel_height == 0:
        return 0.0
    ratio = ref_real_height_m / ref_pixel_height
    return pixel_height * ratio


@com_img_det_retrain.route('/')
def index():
    return render_template('poledetection.html')


@com_img_det_retrain.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(com_img_det_retrain.config['STATIC_FOLDER'], filename)

# NEW — WIRE STRAIGHT / HANGING DETECTION FUNCTIONS
# --------------------------------------------------

def calculate_wire_angle_and_sag(cropped_wire):
    """Returns (angle_deg, sag_amount_pixels)"""
    if cropped_wire is None or cropped_wire.size == 0:
        return 0, 0

    gray = cv2.cvtColor(cropped_wire, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    # 1️⃣ Hough line → angle detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=15)
    angle_deg = 0
    if lines is not None:
        # pick the longest line
        max_len = 0
        best = None
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length > max_len:
                max_len = length
                best = l[0]

        if best is not None:
            x1, y1, x2, y2 = best
            dx = x2 - x1
            dy = y2 - y1
            angle_deg = abs(math.degrees(math.atan2(dy, dx)))

    # 2️⃣ Sag (curvature) detection
    h, w = edges.shape
    ys, xs = np.where(edges > 0)
    sag_amount = 0
    if len(xs) > 0:
        # straight reference line between endpoints
        x_start, x_end = np.min(xs), np.max(xs)
        y_start = np.mean(ys[xs == x_start])
        y_end = np.mean(ys[xs == x_end])

        # line equation distance
        A = y_end - y_start
        B = x_start - x_end
        C = (x_end * y_start) - (x_start * y_end)

        distances = np.abs(A * xs + B * ys + C) / math.sqrt(A*A + B*B)
        sag_amount = float(np.max(distances))

    return angle_deg, sag_amount


def is_wire_sagging(x1, y1, x2, y2, height_threshold_ratio=0.10):
    
    box_height = abs(y2 - y1)
    box_width = abs(x2 - x1)

    # Straight wires → very flat lines → height is very small
    if box_height < box_width * height_threshold_ratio:
        return False  # wire is straight

    return True  # wire sagging



RETRAIN_CONF_THRESHOLD = 0.25

# NEW PREDICT ROUTE — USES ONLY ONE MODEL FOR ALL 3 TASKS

@com_img_det_retrain.route('/predict', methods=['POST'])

def predict():

    # Clear previously uploaded images
    clear_all_image_folders()
    files = request.files.getlist("files")
    if len(files) == 0:
        return jsonify({"error": "No files received by backend"}), 400

    final_results = []
    

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        original_image = cv2.imread(filepath)
        if original_image is None:
            continue

        current_image_to_plot = original_image.copy()
        anomaly = False
        reason_list = []

        # Run unified YOLO model ONCE
        results = model.predict(source=filepath, conf=0.25, iou=0.45, save=False)

        pole_detected = False
        fire_detected = False
        wire_detected = False

        sagging_wire_found = False  
        max_conf_in_image = 0.0

        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                conf_val = float(conf)
                max_conf_in_image = max(max_conf_in_image, conf_val)
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                label_name = model.names[int(cls)]

                # Choose colors by class
                if label_name == "pole":
                    color = (0, 255, 0)
                    pole_detected = True

                elif label_name in ["fire", "smoke", "pole_fire"]:
                    color = (0, 0, 255)
                    fire_detected = True
                    anomaly = True
                    reason_list.append("Fire/Smoke detected")

                # elif label_name in ["wire", "pole_wire"]:
                #     color = (0, 255, 255)
                #     wire_detected = True
                #     anomaly = True
                #     reason_list.append("Wire anomaly detected")
                

                elif label_name in ["wire", "pole_wire"]:
                    if is_wire_sagging(x1, y1, x2, y2):

                        color = (0, 255, 255)
                        wire_detected = True
                        anomaly = True

                        if not sagging_wire_found:
                            reason_list.append("Hanging wire detected")
                            sagging_wire_found = True

                        cv2.rectangle(current_image_to_plot, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(
                            current_image_to_plot,
                            f"{label_name} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )
                    else:
                        continue


                else:
                    color = (255, 255, 255)

                # Draw boxes
                cv2.rectangle(current_image_to_plot, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    current_image_to_plot,
                    f"{label_name} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Only tilt calculation for POLE
                if label_name == "pole":
                    cropped = original_image[y1:y2, x1:x2]
                    tilt = round(calculate_pole_properties(cropped), 2)
                    if not (85 <= tilt <= 95):
                        anomaly = True
                        reason_list.append(f"Pole tilt anomaly: {tilt}°")
        import time
        import shutil

        if max_conf_in_image < RETRAIN_CONF_THRESHOLD:
            timestamp = int(time.time())
            retrain_filename = f"{timestamp}_{filename}"
            retrain_path = os.path.join(RETRAIN_IMAGE_FOLDER, retrain_filename)

            shutil.copy(filepath, retrain_path)

            # Optional metadata logging
            with open(RETRAIN_META, "a") as f:
                f.write(f"{retrain_filename},{max_conf_in_image}\n")

        # FINAL TEXT
        reason = "No anomaly detected" if not anomaly else "; ".join(reason_list)

        # Save original & processed
        original_save_path = os.path.join(ORIGINAL_FOLDER, filename)
        Image.fromarray(original_image[..., ::-1]).save(original_save_path)

        processed_filename = f"processed_{filename}"
        output_path = os.path.join(STATIC_FOLDER, processed_filename)
        Image.fromarray(current_image_to_plot[..., ::-1]).save(output_path)

        final_results.append({
            "name": filename,
            "preview": f"/static/{processed_filename}",
            "preview_original": f"/static/original/{filename}", 
            "anomaly": anomaly,
            "reason": reason,
            "pole_detected": pole_detected,
            "fire_detected": fire_detected,
            "wire_detected": wire_detected,
        })

    return jsonify({"results": final_results})


if __name__ == '__main__':
    com_img_det_retrain.run(host='127.0.0.1', port=5005, debug=True)
