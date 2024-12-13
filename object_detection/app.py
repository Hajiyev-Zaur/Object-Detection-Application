# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO
from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    if file.filename.endswith(('jpg', 'jpeg', 'png')):
    
        image = cv2.imread(file_path)
        if image is None:
            return "Image could not be loaded!", 400

        results = model(image)
        if not results:
            return "The model could not predict!", 500

        annotated_image = results[0].plot()
        output_path = os.path.join("outputs", file.filename)
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(output_path, annotated_image)

        return send_file(output_path, mimetype="image/jpeg")

    elif file.filename.endswith(('mp4', 'avi')):
       video_path = file_path
    output_video_path = os.path.join("outputs", file.filename)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        if results:
            frame = results[0].plot()

        out.write(frame)

    cap.release()
    out.release()

    return send_file(output_video_path, mimetype="video/mp4")


    return "Unsupported file type!", 400

if __name__ == "__main__":
    app.run(debug=True)


