# 🗑️ Real-time Waste Detection using YOLO11s

ระบบตรวจจับและจำแนกประเภทขยะแบบเรียลไทม์ผ่านกล้องเว็บแคม โดยพัฒนาบนสถาปัตยกรรม **YOLO11** (เวอร์ชันล่าสุด) ร่วมกับ **OpenCV** บนภาษา Python เพื่อเพิ่มประสิทธิภาพและความแม่นยำในการคัดแยกขยะ

---

## 🌟 Features (คุณสมบัติระบบ)
*   **Real-time Object Detection:** ตรวจจับและตีกรอบวัตถุขยะผ่านกล้อง Webcam ด้วยความเร็วสูงอัตโนมัติ
*   **Multi-Class Classification:** รองรับการจำแนกประเภทขยะหลัก 3 ประเภทตามมาตรฐาน:
    1.  `General Waste` (ขยะทั่วไป)
    2.  `Hazardous Waste` (ขยะอันตราย)
    3.  `Recycle Waste` (ขยะรีไซเคิล)
*   **Multi-Version Testing:** รองรับการสลับสคริปต์สลับทดสอบประสิทธิภาพโมเดลได้ครบทั้ง 4 เวอร์ชัน เพื่อเปรียบเทียบหาเวอร์ชันที่ดีที่สุดสำหรับนำไปใช้งานจริง

---

## 📁 Dataset & Multi-Version Experiment
ชุดข้อมูล (Dataset) อ้างอิงจากคลังข้อมูลรูปแบบตรวจจับวัตถุของ `WasteD` ผ่านแพลตฟอร์ม Roboflow โดยในการทดลองนี้ได้ทำการเทรนโมเดลเพื่อเปรียบเทียบประสิทธิภาพทั้งหมด **4 เวอร์ชัน (v1 - v4)** ด้วยเงื่อนไขพารามิเตอร์มาตรฐาน:
*   **Image Size (`imgsz`):** 640 x 640 px
*   **Training รอบ (`epochs`):** 50 Epochs per version
*   **Hardware Accelerator:** NVIDIA T4 GPU (Google Colab Environment)

---

## 💻 Tech Stack & Dependencies
*   **Language:** Python
*   **Deep Learning Framework:** `ultralytics` (YOLO11)
*   **Computer Vision library:** `opencv-python`

---

## 🚀 How to Run (วิธีการติดตั้งและรันใช้งาน)

### 1. การเตรียมสภาพแวดล้อมและการติดตั้ง Library
เปิด Terminal ในโฟลเดอร์โปรเจกต์แล้วรันคำสั่งติดตั้ง Dependencies หลัก:
```bash
pip install ultralytics opencv-python
```

### 2. โครงสร้างโฟลเดอร์ในโปรเจกต์
จัดเก็บไฟล์ให้อยู่ในระนาบเดียวกันในโฟลเดอร์หลักเพื่อป้องกันการเรียก Path ผิดพลาด:
```text
Waste_detection/
├── best_v1.pt         # ไฟล์น้ำหนักโมเดล Version 1
├── best_v2.pt         
├── best_v3.pt
├── best_v4.pt
├── Test_came1.py      # สคริปต์เปิดกล้องทดสอบ Version 1
├── Test_came2.py      
├── Test_came3.py      
└── Test_came4.py      
```

### 3. คำสั่งเปิดระบบรันใช้งานกล้องเว็บแคม
คุณสามารถเลือกสั่งรันไฟล์ตามเวอร์ชันโมเดลที่ต้องการทดสอบผ่าน Terminal ได้ทันที:

*   **ทดสอบโมเดล Version 1:**
    ```bash
    python Test_came1.py
    ```
*   **ทดสอบโมเดล Version 4:**
    ```bash
    python Test_came4.py
    ```

💡 **วิธีปิดโปรแกรม:** คลิกที่หน้าต่างกล้องเว็บแคม 1 ครั้ง แล้วกดปุ่มตัว **`q`** บนคีย์บอร์ด หรือกดปุ่มกากบาท `[X]` ระบบจะทำการปิดกล้องและคืนสิทธิ์หน้าจออย่างปลอดภัย

---

## 👩‍💻 Contributors / Credits
*   **Framework:** Ultralytics YOLO11
*   **Dataset Source:** Roboflow (WasteD Dataset)
*   **Project Developer:** [Panatchaya Hoyjan]
