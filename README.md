# NSFW Image Detection and Filtering
Detect image adult content and apply censorship 

# Install
`$ pip install -r requirements.txt`


# NSFW Image detection and filtering

Check these paths:
* Source path: `training_samples/selected/`
* Output path: `training_samples/nsfw_recognized/`

Launch the filter:
`$ python3 nsfw_filter.py`

___

## Main resources:
* [YOLO v11](https://docs.ultralytics.com/it/models/yolo11/)<br>
  Useful for documentation and pre-trained models
* [HuggingFace - erax-ai/EraX-Anti-NSFW-V1.1](https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1?not-for-all-audiences=true)<br>
  The original repo with the script used for the image detection and filtering

## Other resources:
* [Sexy YOLO](https://github.com/algernonx/SexyYolo)<br>
  A hierarchical detection method based on Yolov3, which can work on joint classification and detection dataset, such as COCO and NSFW. So that Yolo could detect coco categories and sexy or porn person simultaneously. (Tensorflow 1.x)
* [🌔 moondream](https://github.com/vikhyat/moondream)<br>
  An AI script for describe image contents
