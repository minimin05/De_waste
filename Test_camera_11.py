import cv2
from ultralytics import YOLO

# ใส่โมเดลเวอชั่นที่เลือก
model = YOLO("best_v01.pt")

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# กำหนดความละเอียดแล้วแต่เรา (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Turning on the webcam......")
print("Click the camera window and press the 'q' key on your keyboard, or click the [X] button")

cv2.namedWindow("YOLO11 Waste Detection", cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ไม่สามารถเปิดกล้องได้")
        break

    # ตรวจจับวัตถุในเฟรมอันนี้ตั้งไว้(conf=0.5)
    results = model(frame, conf=0.5)

    # วาดกรอบและป้ายชื่อคลาสขยะลงบนภาพเพื่อแก้ปัญหา list object
    annotated_frame = results[0].plot()

    # แสดงหน้าจอ
    cv2.imshow("YOLO11 Waste Detection", annotated_frame)

    # ปิดกล้องเมื่อกด 'q' หรือกดกากบาท X ที่มุมหน้าต่างกล้อง
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("YOLO11 Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("ปิดโปรแกรมเรียบร้อยครับ")

#ลองใช้ พิมพ์ python Test_camera.py