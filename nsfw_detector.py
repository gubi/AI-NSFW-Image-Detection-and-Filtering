#!/usr/bin/env python3

# NSFW Image detection and filtering
# ------------------------------------------------------------------------------
# author: Alessandro Gubitosi
# date: 2025-02-05
#
# @see Fine tuning scripts: https://github.com/EraX-JS-Company/EraX-NSFW-V1.0
# @source: https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1?not-for-all-audiences=true

'''
OPTIONAL
Decomment these lines if you need to download pre-trained models
'''
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="erax-ai/EraX-NSFW-V1.0", local_dir="./", force_download=True)

import json
import numpy as np
import os
import supervision as sv
from PIL import Image
from ultralytics import YOLO

def save_image(image, image_path = "training_samples/nsfw_recognized/"):
  # image = Image.fromarray(array)
  Image.save(image_path)
  print("Saved image: {}".format(image_path))

IOU_THRESHOLD        = 0.3
CONFIDENCE_THRESHOLD = 0.2

pretrained_path = "pretrains/erax_nsfw_yolo11m.pt"
# pretrained_path = "erax_nsfw_yolo11n.pt"
# pretrained_path = "erax_nsfw_yolo11s.pt"
model_path = "training_samples/selected/"
# model_path = [
#     "training_samples/selected/1722985818194150.png",
#     "training_samples/selected/1724897142625564.jpg"
# ]
recognized_path = "training_samples/nsfw_recognized/"


model = YOLO(pretrained_path)
results = model(model_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
json_sink = sv.JSONSink(recognized_path + "report.json")

# Save JSON
with json_sink as sink:
    for result in results:
        filename = os.path.basename(result.path)
        recognized_filename = recognized_path + filename
        annotated_image = result.orig_img.copy()
        detected = []

        h, w = annotated_image.shape[:2]
        anchor = h if h > w else w

        detections = sv.Detections.from_ultralytics(result)
        label_annotator = sv.LabelAnnotator(
            text_color = sv.Color.BLACK,
            text_position = sv.Position.CENTER,
            text_scale = anchor/1700
        )

        pixelate_annotator = sv.PixelateAnnotator(pixel_size = anchor/50)

        annotated_image = pixelate_annotator.annotate(
            scene = annotated_image.copy(),
            detections = detections
        )


        for found in detections.data.values():
            detected = str(found)
            detected_count = len(found)
            # print(cn)

        # print(detected)
        # exit()
        annotated_image = label_annotator.annotate(
            annotated_image,
            detections = detections
        )

        sink.append(
            detections,
            custom_data = {
                "filename": filename,
                "detections": {
                    "count": detected_count,
                    "items": detected
                }
            }
        )
        pilimg = sv.cv2_to_pillow(annotated_image)
        pilimg.save(recognized_filename)
    # sv.plot_image(annotated_image, size=(10, 10))
    # save_image(annotated_image, recognized_filename)
