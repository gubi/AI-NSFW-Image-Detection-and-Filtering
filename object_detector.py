from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import logging

IOU_THRESHOLD        = 0.3
CONFIDENCE_THRESHOLD = 0.2

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", passes it
    through YOLOv11 object detection network and returns and array
    of bounding boxes.
    :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv11 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """

    # Load a model
    # model = YOLO("./yolo11n.yaml")  # build a new model from YAML
    # model = YOLO("./yolo11n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # model = YOLO("best.pt")
    # model = YOLO("yolo11n.pt")
    # model = YOLO("./yolo11n.yaml").load("erax_nsfw_yolo11n.pt")
    # model = YOLO("./yolo11n.yaml").load("yolo11n.pt").load("erax_nsfw_yolo11n.pt")
    # model = YOLO("SexyCocoModel/yolov3.data-00000-of-00001")

    # See https://huggingface.co/erax-ai/EraX-NSFW-V1.0/blob/main/erax_nsfw_yolo11n.pt
    model = YOLO("pretrains/yolo11m.pt")
    model = YOLO("pretrains/erax_nsfw_yolo11m.pt")

    # model1 = YOLO("pretrains/yolo11m.pt")  # load a pretrained model (recommended for training)
    # model2 = YOLO("datasets/yolo11-seg.yaml")  # build a new model from YAML
    # # model1 = YOLO("pretrains/yolo11m-seg.pt")  # load a pretrained model (recommended for training)
    # # model2 = YOLO("pretrains/erax_nsfw_yolo11m.pt") #.load("pretrains/yolo11m.pt")  # load a pretrained model (recommended for training)
    # # model = YOLO("datasets/yolo11-seg.yaml") #.load("yolo11.pt")  # build from YAML and transfer weights
    #
    # # See https://github.com/ultralytics/ultralytics/issues/5882#issuecomment-2188576650
    # # Extract class names from the dictionaries
    # classes1 = list(model1.names.values())
    # classes2 = list(model2.names.values())
    #
    # # Combine class labels
    # combined_classes = classes1 + [cls for cls in classes2 if cls not in classes1]
    #
    # # Initialize a new YOLOv8 model with the architecture of one of the pretrained models
    # combined_model = YOLO('pretrains/yolo11m.pt')
    # combined_model.model.nc = len(combined_classes)
    # combined_model.model.names = {i: name for i, name in enumerate(combined_classes)}
    #
    # # Transfer weights
    # weights1 = model1.model.state_dict()
    # weights2 = model2.model.state_dict()
    # combined_weights = combined_model.model.state_dict()
    #
    # for key in weights1:
    #     if key in combined_weights:
    #         combined_weights[key] = weights1[key]
    # for key in weights2:
    #     if key in combined_weights:
    #         combined_weights[key] = weights2[key]
    #
    # combined_model.model.load_state_dict(combined_weights, strict=False)
    #
    # # Save the combined model
    # # torch.save(combined_model.model.state_dict(), 'yolov8_combined_model.pt')
    #
    # # Perform inference with the combined model
    # # results = combined_model(source="video.mp4", show=True, conf=0.1, save=True)
    # # print(results)
    #
    # results = combined_model.predict(Image.open(buf), conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
    results = model.train(data="coco8.yaml", epochs=2)
    # result = results[0]
    print(results)
    # output = []
    # for box in result.boxes:
    #     x1, y1, x2, y2 = [
    #         round(x) for x in box.xyxy[0].tolist()
    #     ]
    #     class_id = box.cls[0].item()
    #     prob = round(box.conf[0].item(), 2)
    #     output.append([
    #         x1, y1, x2, y2, result.names[class_id], prob
    #     ])
    # # print(jsonify(result))
    # return output

serve(app, host='127.0.0.1', port=3000)
