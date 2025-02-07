#!/usr/bin/env python3

# NSFW Image content detection
# ------------------------------------------------------------------------------
# author: Alessandro Gubitosi
# date: 2025-02-07
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

# Common variables
# --------------------------------------------------------------------------
IOU_THRESHOLD        = 0.3
CONFIDENCE_THRESHOLD = 0.1
# Key Name: cosmic-bear-616
MOONDREAM_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJkZDkyM2U2Mi1iZmE4LTRhMzUtYmFmYS02MjM4NmQ0ZTAwNTIiLCJpYXQiOjE3Mzg5MDA3MTF9.TzMA1xBoGpUthKdU8tQuIrfOpAvOR_SMIB2Sb_oUVcs"

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

# NSFW image filtering script
# --------------------------------------------------------------------------
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

# Get image description function
# --------------------------------------------------------------------------
moondream_model = md.vl(api_key=MOONDREAM_API_KEY)
moondream_query = """
Give me a brief description of this image, and a detailed comma-separated list of unique objects visible in this image, no general terms.
Output JSON object with keys:
- 'subject': (string) An objective description of the subject in the image in 50 characters
- 'short': (string) a short description for image label (max 25 words)
- 'long': (string) a long description, max 250 characters
- 'tags': (array) a list of univocal and not repeated 6 tags
"""

# Generate an AI image description
## @params <string>                 img                   The image to process
def get_image_description(img):
    image = Image.open(img)
    #encoded_image = moondream_model.encode_image(image)
    description = {}
    
    try:
        answer = moondream_model.query(image, moondream_query)["answer"]
        description = json.loads(answer)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}, response: {answer}")
        description = {  # provide default values if JSON decode fails
            "subject": None,
            "short": None,
            "long": None,
            "tags": []
        }
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        description = {  # provide default values if any other error occurs
            "subject": None,
            "short": None,
            "long": None,
            "tags": []
        }
    
    description = {
        "subject": description.get("subject"),
        "short": description.get("short"),
        "long": description.get("long"),
        "tags": list(set(description.get("tags")))
    }
    return description

# Detect function
# --------------------------------------------------------------------------
# Perform a detection of a selected pretrained model on given source
# The source can be both a string or array
def detect(pretrained, source, sex_check = False):
    detect = []
    model = YOLO(pretrained)
    results = model(source, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
    
    
    for result in results:
        file = os.path.basename(result.path)
        recognized_filename = recognized_path + file
        annotated_image = result.orig_img.copy()
        
        detections = sv.Detections.from_ultralytics(result)
        for found in detections.data.values():
            detected = found.tolist()
            detected_count = len(found)
        
        if sex_check and detected_count > 0:
            has_sex = True
        else:
            has_sex = False
        
        custom_data = {
            "file": file,
            "path": recognized_path,
            "has_sex": has_sex,
            "description": get_image_description(result.path),
            "detections": {
                "count": detected_count,
                "items": detected
            }
        }
        detect.append(custom_data)
    return detect

# Merge detections function
# --------------------------------------------------------------------------
subject_path = source_2_paths

default_detect = detect(yolo_pretrained_pt, subject_path)
nsfw_detect = detect(nsfw_pretrained_pt, subject_path, True)

def merge_detections(lst1, lst2):
    merged_list = []  # Create a list to store merged results
    for def_item, nsfw_item in zip(lst1, lst2):  # Assuming lst1 and lst2 have the same length and corresponding items
        # Use square bracket notation to access dictionary keys
        file = def_item.get("file")
        path = def_item.get("path")
        
        has_sex = def_item.get("has_sex") or nsfw_item.get("has_sex")
        count = def_item.get("detections", {}).get("count", 0) + nsfw_item.get("detections", {}).get("count", 0)
        items = def_item.get("detections", {}).get("items", []) + nsfw_item.get("detections", {}).get("items", [])
        result = dict((i, items.count(i)) for i in items)

        description = def_item.get("description")
        
        merged_item = {
            "file": file,
            "path": path,
            "has_sex": has_sex,
            "description": description,
            "detections": {
                "count": count,
                "items": result
            }
        }
        merged_list.append(merged_item)
    return json.dumps(merged_list)

print()
print("final", merge_detections(default_detect, nsfw_detect))
