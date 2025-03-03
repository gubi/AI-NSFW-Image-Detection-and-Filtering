# AI NSFW Image Detection and Filtering
![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/gubi/NSFW-Image-Detection-and-Filtering/blob/master/training_samples/nsfw_recognized/1724897142625564.jpg)

Detect image adult content and apply censorship using AI

&nbsp;
&nbsp;

### Install requirements
Main<br>
`$ pip install -r requirements.txt`

&nbsp;

### Execution
> #### Keep in mind these folders
> * Source path: `training_samples/selected/`
> * NSFW filtered images path: `training_samples/nsfw_recognized/`


&nbsp;

### Launch the filter
`$ python3 nsfw_filter.py`

### Execute the image detector
`$ python3 image_detector`

&nbsp;

___

#### Main resources:
* [YOLO v11](https://docs.ultralytics.com/it/models/yolo11/)<br>
  Useful for documentation and pre-trained models
* [HuggingFace - erax-ai/EraX-Anti-NSFW-V1.1](https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1?not-for-all-audiences=true)<br>
  The original repo with the script used for the image detection and filtering
* [🌔 moondream](https://github.com/vikhyat/moondream)<br>
  An AI script for describe image contents

#### Other resources:
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
