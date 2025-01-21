from ultralytics import YOLO

model = YOLO("best.pt")

model.export(format="ncnn", imgsz=320)

ncnn_model = YOLO("yolobox_ncnn_model")

results = ncnn_model("https://ultralytics.com/images/bus.jpg")
