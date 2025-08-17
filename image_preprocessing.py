from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n-seg.pt")  # segmentation version

print("Model loaded.")
img = cv2.imread("car.jpg")
res = model(img)[0]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
for m, cls in zip(res.masks.xy, res.boxes.cls.cpu().numpy()):
    if int(cls) == 2:  # 'car'
        pts = np.int32([m])
        cv2.fillPoly(mask, pts, 255)

white_bg = 255 * np.ones_like(img, dtype=np.uint8)
final = np.where(mask[..., None] == 255, img, white_bg)

cv2.imwrite("car_whitebg.jpg", final)
print("Saved car_whitebg.jpg")
