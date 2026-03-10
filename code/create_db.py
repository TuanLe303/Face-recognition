import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# --- Cấu hình ---
DATASET_DIR = r"C:\1MATERIAL\3LHMT\FPT\CPV301\assignment\dataset"
DB_PATH     = r"C:\1MATERIAL\3LHMT\FPT\CPV301\assignment\db\face_db.pkl"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}

# Khởi tạo model InsightFace
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

face_db: dict[str, list[np.ndarray]] = {}

# --- Quét dataset theo cấu trúc: dataset/<TenNguoi>/<anh.jpg> ---
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue  # bỏ qua file lẻ ở thư mục gốc

    print(f"\n[{person_name}]")
    for fname in os.listdir(person_dir):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue

        img_path = os.path.join(person_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] Không đọc được: {fname}")
            continue

        faces = app.get(frame)
        if not faces:
            print(f"  [WARN] Không phát hiện khuôn mặt: {fname}")
            continue

        embedding: np.ndarray = faces[0].embedding
        face_db.setdefault(person_name, []).append(embedding)
        print(f"  [OK]  {fname}")

# --- Lưu database ---
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
with open(DB_PATH, "wb") as f:
    pickle.dump(face_db, f)

print(f"\n=== Database đã lưu tại: {DB_PATH} ===")
for name, vecs in face_db.items():
    print(f"  {name}: {len(vecs)} ảnh")
