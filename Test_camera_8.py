import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล (ตรวจสอบให้มั่นใจว่าใส่ชื่อไฟล์โมเดล YOLOv8 ของคุณถูกต้อง)
model = YOLO("best_v4_8.pt")  

# 2. ตั้งค่าเปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("กำลังเปิดกล้องเว็บแคม (YOLOv8)...")

# 📌 จุดแก้ไขที่ 1: เปลี่ยนหัวข้อหน้าต่าง Windows เป็น YOLOv8
cv2.namedWindow("YOLOv8 Waste Detection", cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ไม่สามารถเปิดกล้องได้")
        break

    # 3. สั่งตรวจจับวัตถุด้วยโมเดล YOLOv8
    results = model(frame, conf=0.5)[0]  # เติม [0] เพื่อแก้ปัญหา list object

    # 4. วาดกรอบและป้ายชื่อคลาสขยะลงบนภาพ
    annotated_frame = results.plot()

    # 📌 จุดแก้ไขที่ 2: เปลี่ยนชื่อหน้าต่างตอนโชว์ภาพให้ตรงกันเป๊ะเป็น YOLOv8
    cv2.imshow("YOLOv8 Waste Detection", annotated_frame)

    # 📌 จุดแก้ไขที่ 3: แก้ไขตัวตรวจสอบการปิดหน้าต่างให้ชื่อตรงกัน
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("YOLOv8 Waste Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("ปิดโปรแกรมเรียบร้อยครับ")
