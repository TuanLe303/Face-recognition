import os
import cv2
import pickle
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
import threading
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ==========================================
# KHỞI TẠO MODEL
# ==========================================
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# BƯỚC 2.1: TẢI DATABASE VÀO RAM
# ==========================================
DB_PATH     = r"C:\1MATERIAL\3LHMT\FPT\CPV301\assignment\db\face_db.pkl"
DATASET_DIR = r"C:\1MATERIAL\3LHMT\FPT\CPV301\assignment\dataset"
with open(DB_PATH, "rb") as f:
    database_vectors: dict = pickle.load(f)
print(f"Đã tải database: {list(database_vectors.keys())}")

# ==========================================
# BƯỚC 2.2: HÀM COSINE SIMILARITY
# ==========================================
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính độ tương đồng cosine giữa 2 vector. Kết quả trong [-1, 1], càng gần 1 càng giống."""
    dot   = np.dot(vec_a, vec_b)
    norm  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm) if norm != 0 else 0.0

THRESHOLD = 0.45

# ==========================================
# CẤU HÌNH CHẾ ĐỘ ĐĂNG KÝ NGƯỜI MỚI
# ==========================================
ANGLE_PROMPTS = [
    "Buoc 1/5: Nhin THANG vao camera",
    "Buoc 2/5: Quay mat sang TRAI",
    "Buoc 3/5: Quay mat sang PHAI",
    "Buoc 4/5: NGUA dau len tren",
    "Buoc 5/5: CUI dau xuong duoi",
]

MODE_RECOGNITION = "recognition"
MODE_ENROLLMENT  = "enrollment"
mode                  = MODE_RECOGNITION
enrollment_name       = ""
enrollment_embeddings = []
enrollment_step       = 0


def save_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(database_vectors, f)

class RTSPVideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Trả về tuple (ret, frame) để tương thích 100% với code hiện tại của bạn
        return (self.grabbed, self.frame)

    def isOpened(self):
        return self.stream.isOpened()

    def stop(self):
        self.stopped = True

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "url"))
rtsp_url = os.getenv("RTSP_URL", "0")
if rtsp_url == "0":
    rtsp_url = 0
    print("using laptop_cam")
else:
    print("using IP camera")
cap = RTSPVideoStream(rtsp_url).start()


if not cap.isOpened():
    print("Loi: Khong the ket noi den Camera.")
    exit()

print("He thong dang chay...")
print("  Nhan 'n' de dang ky nguoi moi")
print("  Nhan 'q' de thoat")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Canh bao: Mat tin hieu tu camera.")
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (720, 1280))

    display = frame.copy()

    # ==========================================
    # CHE DO NHAN DIEN BINH THUONG
    # ==========================================
    if mode == MODE_RECOGNITION:
        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            current_embedding = face.embedding
            best_name, best_score = "Unknown", 0.0

            for person_name, vectors in database_vectors.items():
                for db_vec in vectors:
                    score = cosine_similarity(current_embedding, db_vec)
                    if score > best_score:
                        best_score = score
                        best_name  = person_name

            if best_score >= THRESHOLD:
                label, color = f"{best_name} ({best_score:.2f})", (0, 255, 0)
            else:
                label, color = f"Unknown ({best_score:.2f})", (0, 0, 255)

            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(display, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if face.kps is not None:
                for pt in face.kps.astype(int):
                    cv2.circle(display, (pt[0], pt[1]), 2, (0, 255, 255), -1)

        cv2.putText(display, "[n] Dang ky nguoi moi  |  [q] Thoat",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ==========================================
    # CHE DO DANG KY NGUOI MOI
    # ==========================================
    elif mode == MODE_ENROLLMENT:
        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 215, 255), 2)

        # Thanh thong bao nền tối phía trên
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 135), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        cv2.putText(display, f"DANG KY: {enrollment_name}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2)
        cv2.putText(display, ANGLE_PROMPTS[enrollment_step],
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Các chấm tiến trình (progress dots)
        for i in range(5):
            clr = (0, 220, 0) if i < enrollment_step else (70, 70, 70)
            cv2.circle(display, (20 + i * 35, 115), 12, clr, -1)
            cv2.putText(display, str(i + 1), (14 + i * 35, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        cv2.putText(display, "[c] Chup anh  |  [ESC] Huy dang ky",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Face Recognition", display)
    key = cv2.waitKey(1) & 0xFF

    # ==========================================
    # XU LY PHIM BAM
    # ==========================================
    if key == ord('q'):
        break

    elif key == ord('n') and mode == MODE_RECOGNITION:
        # Hien thi thong bao tren man hinh roi hoi ten qua terminal
        msg_frame = display.copy()
        cv2.putText(msg_frame, "Nhap ten/Ma SV trong terminal...",
                    (10, display.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
        cv2.imshow("Face Recognition", msg_frame)
        cv2.waitKey(200)

        print("\n" + "=" * 45)
        name = input("  >> Nhap ten / Ma SV cua nguoi moi: ").strip()
        print("=" * 45)

        if name:
            enrollment_name       = name
            enrollment_embeddings = []
            enrollment_step       = 0
            mode                  = MODE_ENROLLMENT
            print(f"[INFO] Bat dau dang ky cho: {enrollment_name}")
            print("[INFO] Bam 'c' de chup anh theo tung goc nhu huong dan tren man hinh.")
        else:
            print("[WARN] Ten trong, huy dang ky.")

    elif key == ord('c') and mode == MODE_ENROLLMENT:
        faces = app.get(frame)
        if not faces:
            print("[CANH BAO] Khong phat hien khuon mat! Hay chinh lai vi tri va bam 'c' lai.")
        else:
            face      = faces[0]
            embedding = face.embedding
            enrollment_embeddings.append(embedding)

            # Luu anh goc vao dataset/<ten>/ de backup
            person_dir = os.path.join(DATASET_DIR, enrollment_name)
            os.makedirs(person_dir, exist_ok=True)
            img_path = os.path.join(person_dir, f"angle_{enrollment_step}.jpg")
            cv2.imwrite(img_path, frame)

            print(f"[OK] Goc {enrollment_step + 1}/5 - Da luu: {img_path}")
            enrollment_step += 1

            # Flash vien xanh bao da chup thanh cong
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (0, 255, 0), 10)
            cv2.putText(flash, "Da chup!",
                        (flash.shape[1] // 2 - 90, flash.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
            cv2.imshow("Face Recognition", flash)
            cv2.waitKey(700)

            if enrollment_step >= 5:
                # Cap nhat RAM va luu file pkl
                database_vectors[enrollment_name] = enrollment_embeddings
                save_database()
                print(f"\n[THANH CONG] Da dang ky '{enrollment_name}' voi {len(enrollment_embeddings)} embedding.")
                print(f"[INFO] Database hien tai: {list(database_vectors.keys())}")

                # Thong bao hoan thanh tren man hinh
                done_frame = frame.copy()
                cv2.putText(done_frame, f"Dang ky thanh cong: {enrollment_name}",
                            (10, done_frame.shape[0] // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(done_frame, "Tro ve che do nhan dien...",
                            (10, done_frame.shape[0] // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imshow("Face Recognition", done_frame)
                cv2.waitKey(2000)

                # Reset ve che do nhan dien
                mode                  = MODE_RECOGNITION
                enrollment_name       = ""
                enrollment_embeddings = []
                enrollment_step       = 0

    elif key == 27 and mode == MODE_ENROLLMENT:  # ESC
        print("[HUY] Da huy dang ky.")
        mode                  = MODE_RECOGNITION
        enrollment_name       = ""
        enrollment_embeddings = []
        enrollment_step       = 0

# Don dep bo nho khi ket thuc
cap.stop()
cv2.destroyAllWindows()