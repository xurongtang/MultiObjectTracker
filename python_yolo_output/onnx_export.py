from ultralytics import YOLO

# model = YOLO("/home/rton/MultiObjectTracker/python_yolo_output/yolo12n.pt")  # 或 yolov5m, yolov5l 等
# model.export(format="onnx", imgsz=640, opset=12)

# import cv2
# net = cv2.dnn.readNet("yolo12n.onnx")
# print("Output layers:", net.getUnconnectedOutLayersNames())

model = YOLO(model='/home/rton/MultiObjectTracker/python_yolo_output/yolo12n.pt')
res = model.predict(source="/home/rton/MultiObjectTracker/src/yolo/test.jpeg",imgsz=640)[0]
# res.show()
res.save(filename="res.jpg")  # 将结果保存为res.jpg文件
