from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

model.train(
    data='multi_dataset/data.yaml', 
    epochs=30,
    imgsz=640,
    batch=16,
    name='train_new' 
)
