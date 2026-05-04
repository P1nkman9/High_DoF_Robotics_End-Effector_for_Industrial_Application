from ultralytics import YOLO

model = YOLO('yolov8n-pt')
data_path = "/content/drive/MyDrive/datasets/defect/dataset-yaml"

model.train(
     data=data_path, 
     epochs=200, 
     patience=30, 
     imgsz=1280,
     batch=16, 
     workers=8,

     optimizer='AdamW',
     lro=3.8e-05,
     lrf=0.05,
     momentum=0.8917, 
     weight_decay=2.03e-4,
     
     mixup=0.08, 
     scale=0.7267, 
     degrees=0.612, 
     flipud=0.06, 
     fliplr=0.27, 
     hsv_h=0.0136, 
     hsv_s=0.505, 
     hsv_v=0.1,
     project='/content/drive/MyDrive/datasets/defect/runs/detect', 
     name='test2',
     save=True, 
     save_period=-1, 
     pretrained=True, 
     val=True, 
     plots=True
)