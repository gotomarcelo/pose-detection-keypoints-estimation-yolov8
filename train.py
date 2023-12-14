from ultralytics import YOLO


model = YOLO('yolov8n-pose.yaml') # build a new model from YAML
model = YOLO('yolov8n-pose.pt') # load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt') # build from YAML and transfer weights

model.train(data='C:/Users/marcelo.goto/Desktop/yolo-keypoints/pose-detection-keypoints-estimation-yolov8/config.yaml', epochs=1, imgsz=640 )
