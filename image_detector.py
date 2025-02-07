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
moondream_query = """
Give me a brief description of this image, and a detailed comma-separated list of unique objects visible in this image, no general terms.
Output JSON object with keys:
- 'subject': (string) An objective description of the subject in the image in 50 characters
- 'short': (string) a short subtitle label for image in 10 words
- 'long': (string) a discursive, long detailed description on image content, 250 characters
- 'tags': (array) a list of univocal and not repeated 1 to 6 snake_case tags
"""

# Set to "./" for local execution
drive_prefix = "drive/MyDrive/Colab Notebooks/"

yolo_pretrained_pt = drive_prefix + "pretrains/yolo11m.pt"
nsfw_pretrained_pt = drive_prefix + "pretrains/erax_nsfw_yolo11m.pt"
# nsfw_pretrained_pt = drive_prefix + "erax_nsfw_yolo11n.pt"
# nsfw_pretrained_pt = drive_prefix + "erax_nsfw_yolo11s.pt"
visionx_pretrained_pt = drive_prefix + "pretrains/visionx.pt"

source_path = drive_prefix + "training_samples/selected"
source_path_single_not_sex = drive_prefix + "training_samples/selected/1723652428084736.jpg"
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
test_path =  drive_prefix + "training_samples/test/"

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
        #print(f"Error decoding JSON: {e}, response: {answer}")
        description = {  # provide default values if JSON decode fails
            "subject": "",
            "short": "",
            "long": "",
            "tags": []
        }
    except Exception as e:
        #print(f"An unexpected error occurred: {e}")
        description = {  # provide default values if any other error occurs
            "subject": "",
            "short": "",
            "long": "",
            "tags": []
        }

    description = {
        "subject": description.get("subject"),
        "short": description.get("short"),
        "long": description.get("long"),
        "tags": list(set(description.get("tags")))
    }
    return description


# Gender and Age Detection program by Mahesh Sawant
# source: https://github.com/smahesh29/Gender-and-Age-Detection/tree/master

import cv2
import math
##import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def detect_gender_age(image_path):
    faceProto = test_path + "opencv_face_detector.pbtxt"
    faceModel = test_path + "opencv_face_detector_uint8.pb"
    ageProto = test_path + "age_deploy.prototxt"
    ageModel = test_path + "age_net.caffemodel"
    genderProto = test_path + "gender_deploy.prototxt"
    genderModel = test_path + "gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male','Female']

    faceNet = cv2.dnn.readNet(faceModel,faceProto)
    ageNet = cv2.dnn.readNet(ageModel,ageProto)
    genderNet = cv2.dnn.readNet(genderModel,genderProto)

    video = cv2.VideoCapture(image_path)
    padding = 20
    while cv2.waitKey(1)<0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            #print("No face detected")
            break
        else:
            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                          min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                          :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]

                return {
                    "gender": gender,
                    "age": age
                }

                #cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                #cv2.imshow("Detecting age and gender", resultImg)

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
        
        gender_age = detect_gender_age(filename)
        if gender_age is None:
            gender_age = {}
        
        merged_item = {
            "file": file,
            "path": path,
            "has_sex": has_sex,
            "description": description,
            "detections": {
                "count": count,
                "items": result,
                "human_data": gender_age,
            }
        }
        merged_list.append(merged_item)
    return json.dumps(merged_list)

print("Results")
print(merge_detections(default_detect, nsfw_detect))
