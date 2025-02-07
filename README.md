# NSFW Image Detection and Filtering
Detect image adult content and apply censorship 

## Requirements
Main<br>
`$ pip install -r requirements.txt`

Torch<br>
`!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/index.html`

## NSFW Image detection and filtering

Check these paths:
* Source path: `training_samples/selected/`
* Output path: `training_samples/nsfw_recognized/`

Launch the filter:
`$ python3 nsfw_filter.py`

___

### Main resources:
* [YOLO v11](https://docs.ultralytics.com/it/models/yolo11/)<br>
  Useful for documentation and pre-trained models
* [HuggingFace - erax-ai/EraX-Anti-NSFW-V1.1](https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1?not-for-all-audiences=true)<br>
  The original repo with the script used for the image detection and filtering
* [🌔 moondream](https://github.com/vikhyat/moondream)<br>
  An AI script for describe image contents

### Other resources:
* [Sexy YOLO](https://github.com/algernonx/SexyYolo)<br>
  A hierarchical detection method based on Yolov3, which can work on joint classification and detection dataset, such as COCO and NSFW. So that Yolo could detect coco categories and sexy or porn person simultaneously. (Tensorflow 1.x)
* [Rude Carnie: Age and Gender Deep Learning with TensorFlow](https://github.com/dpressel/rude-carnie)<br>
  Do face detection and age and gender classification on pictures
* [Papers and Codes](https://paperswithcode.com/task/pornography-detection)<br>
  The mission of Papers with Code is to create a free and open resource with Machine Learning papers, code, datasets, methods and evaluation tables
* [NSFWJS](https://github.com/infinitered/nsfwjs)<br>
  NSFW detection on the client-side via TensorFlow.js - Client-side indecent content checking
* [Yahoo OpenNSFW](https://github.com/yahoo/open_nsfw)<br>
  Not Suitable for Work (NSFW) classification using deep neural network Caffe models
* [Awesome - Deepfake / Porn Detection using Deep Learning](https://github.com/subinium/awesome-deepfake-porn-detection)<br>
  A list of useful repositories on Deepfake and porn detection using deep learning
