# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

model = YOLO("yolov8n.pt")

image_folder = "data/coco/images/val2017"
output_folder = "outputs/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]

print(f"A total of {len(image_paths)} images were found.")

for idx, image_path in enumerate(image_paths):
    print(f"{idx+1}. Process the image: {image_path}")
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"{image_path} Failed to read successfully!")
        continue

    # Tahmin yap
    results = model(image)

    if not results:
        print("Prediction results are empty!")
        continue

    annotated_image = results[0].plot()

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Processed Image {idx+1}")
    plt.show()

print(f"All images were successfully processed and output {output_folder}")





