#!/usr/bin/env python3

# NSFW Image detection and filtering
# ------------------------------------------------------------------------------
# author: Alessandro Gubitosi
# date: 2025-02-05
#
# @see Fine tuning scripts: https://github.com/EraX-JS-Company/EraX-NSFW-V1.0
# @source: https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1?not-for-all-audiences=true

import json
import numpy as np
import os
import re
import supervision as sv
from PIL import Image
from ultralytics import YOLO
import moondream as md
import torch

IOU_THRESHOLD        = 0.3
CONFIDENCE_THRESHOLD = 0.1

drive_prefix = "drive/MyDrive/Colab Notebooks/"

yolo_pretrained_pt = drive_prefix + "pretrains/yolo11m.pt"
nsfw_pretrained_pt = drive_prefix + "pretrains/erax_nsfw_yolo11m.pt"
# nsfw_pretrained_pt = drive_prefix + "erax_nsfw_yolo11n.pt"
# nsfw_pretrained_pt = drive_prefix + "erax_nsfw_yolo11s.pt"
visionx_pretrained_pt = drive_prefix + "pretrains/visionx.pt"

source_path = drive_prefix + "training_samples/selected"
source_path_single_not_sex = drive_prefix + "training_samples/selected/1698797738754547.jpg"
source_2_paths = [
    drive_prefix + "training_samples/selected/1722985818194150.png",
    drive_prefix + "training_samples/selected/1724897142625564.jpg"
]
source_4_paths = [
    drive_prefix + "training_samples/selected/1698797738754547.jpg",
    drive_prefix + "training_samples/selected/1722985818194150.png",
    drive_prefix + "training_samples/selected/1724579240484244.jpg",
    drive_prefix + "training_samples/selected/1721751848855669.jpg"
]
recognized_path = drive_prefix + "training_samples/nsfw_recognized/"



def filter_nsfw(pretrained, source):
    model = YOLO(pretrained)
    results = model(source, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
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

            # Draw labels
            # label_annotator = sv.LabelAnnotator(
            #     text_color = sv.Color.BLACK,
            #     text_position = sv.Position.CENTER,
            #     text_scale = anchor/1700
            # )

            # Pixelate image
            pixelate_annotator = sv.PixelateAnnotator(pixel_size = anchor/50)
            annotated_image = pixelate_annotator.annotate(
                scene = annotated_image.copy(),
                detections = detections
            )

            # Add labels to image
            # annotated_image = label_annotator.annotate(
            #     annotated_image,
            #     detections = detections
            # )

            for found in detections.data.values():
                detected = str(found)
                detected_count = len(found)

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

filter_nsfw(nsfw_pretrained_pt, source_path)
