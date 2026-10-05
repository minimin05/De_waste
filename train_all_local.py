import os
import zipfile
from ultralytics import YOLO

# func แตกไฟล์ ZIP อัตโนมัติในคอม
def unzip_file(zip_name, extract_to):
    if not os.path.exists(extract_to):
        print(f" กำลังแตกไฟล์ {zip_name}...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f" แตกไฟล์ {zip_name} สำเร็จ!")
    else:
        print(f" พบโฟลเดอร์ {extract_to} อยู่แล้ว ข้ามการแตกไฟล์")

# func find data.yaml in folder
def find_yaml(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'data.yaml' in files:
            return os.path.join(root, 'data.yaml')
    return None

if __name__ == '__main__':
    # แยกไฟล์ ซิป ทั้ง 4 เวอ
    files_in_dir = os.listdir('.')
    
    zip_v1 = [f for f in files_in_dir if f.startswith('WasteD.v1') and f.endswith('.zip')]
    zip_v2 = [f for f in files_in_dir if f.startswith('WasteD.v2') and f.endswith('.zip')]
    zip_v3 = [f for f in files_in_dir if f.startswith('WasteD.v3') and f.endswith('.zip')]
    zip_v4 = [f for f in files_in_dir if f.startswith('WasteD.v4') and f.endswith('.zip')]

    if zip_v1: unzip_file(zip_v1, "v1_dataset")
    if zip_v2: unzip_file(zip_v2, "v2_dataset")
    if zip_v3: unzip_file(zip_v3, "v3_dataset")
    if zip_v4: unzip_file(zip_v4, "v4_dataset")

    # หาตำแหน่งไฟล์ yaml จริง
    yaml_v1 = find_yaml("v1_dataset")
    yaml_v2 = find_yaml("v2_dataset")
    yaml_v3 = find_yaml("v3_dataset")
    yaml_v4 = find_yaml("v4_dataset")

    # ตั้งไว้ว่าให้ เทรน 10 รอบ
    # ver 1
    if yaml_v1:
        print("\n [1/4] --- เริ่มเทรน Version 1 (10 Epochs) ---")
        model = YOLO("yolo11s.pt")
        model.train(data=yaml_v1, epochs=10, imgsz=640, workers=2, device='cpu', project="local_waste_runs", name="v1_model")

    # ver 2
    if yaml_v2:
        print("\n [2/4] --- เริ่มเทรน Version 2 (10 Epochs) ---")
        model = YOLO("yolo11s.pt")
        model.train(data=yaml_v2, epochs=10, imgsz=640, workers=2, device='cpu', project="local_waste_runs", name="v2_model")

    # ver 3
    if yaml_v3:
        print("\n [3/4] --- เริ่มเทรน Version 3 (10 Epochs) ---")
        model = YOLO("yolo11s.pt")
        model.train(data=yaml_v3, epochs=10, imgsz=640, workers=2, device='cpu', project="local_waste_runs", name="v3_model")

    # ver 4
    if yaml_v4:
        print("\n [4/4] --- เริ่มเทรน Version 4 (10 Epochs) ---")
        model = YOLO("yolo11s.pt")
        model.train(data=yaml_v4, epochs=10, imgsz=640, workers=2, device='cpu', project="local_waste_runs", name="v4_model")

    print("\n สำเร็จเสร็จสิ้น! โมเดลทั้ง 4 เวอร์ชันเทรนใหม่เวอร์ชันสปีด 10 รอบเรียบร้อยแล้ว!")
