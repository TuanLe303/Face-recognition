"""
app.py – Streamlit-based Face Recognition System.

Pipeline:
    1. Capture frame from webcam / RTSP stream
    2. Pre-process  (CLAHE – adaptive contrast enhancement on L channel)
    3. Detect faces (RetinaFace via InsightFace buffalo_l)
    4. Extract 512-d embedding per face (ArcFace)
    5. Match against database via vectorized cosine similarity
    6. Display results with bounding boxes + name labels

Run with:
    streamlit run code/app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pickle
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis
import time


# ───────────────────────── Constants ───────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_PATH     = os.path.join(BASE_DIR, "db", "face_db.pkl")
THRESHOLD   = 0.5
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}

# How often to run face detection (every Nth frame).
DETECT_EVERY_N = 3

ANGLE_PROMPTS = [
    "Step 1/5: Look **directly** into the camera",
    "Step 2/5: Turn your face to the **left**",
    "Step 3/5: Turn your face to the **right**",
    "Step 4/5: Tilt your head **upwards**",
    "Step 5/5: Tilt your head **downwards**",
]


# ───────────────────────── Pre-processing ──────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(img: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L
    của không gian LAB. Cân bằng sáng cục bộ mà không bị ảnh hưởng bởi
    background, tốt hơn min-max normalize toàn frame."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


st.set_page_config(page_title="Face Recognition", layout="wide")

@st.cache_resource
def load_model():
    model = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.prepare(ctx_id=0, det_size=(320, 320))
    return model

model = load_model()

if "cap" not in st.session_state:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    st.session_state.cap = cap

if "database" not in st.session_state:
    try:
        with open(DB_PATH, "rb") as f:
            st.session_state.database = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        st.session_state.database = {}

if "mode" not in st.session_state:
    st.session_state.mode = "recognition"
    st.session_state.enroll_name = ""
    st.session_state.enroll_step = 0
    st.session_state.enroll_embeddings = []


# ───────────────────────── Index helpers ───────────────────────
def rebuild_index():
    """Pre-compute a matrix of L2-normalized embeddings for fast cosine
    similarity lookup via a single matrix–vector multiplication."""
    db = st.session_state.database
    names, vecs = [], []
    for name, embeddings in db.items():
        for v in embeddings:
            names.append(name)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
    if vecs:
        st.session_state.db_matrix = np.stack(vecs).astype(np.float32)
    else:
        st.session_state.db_matrix = None
    st.session_state.db_names = names


if "db_matrix" not in st.session_state:
    rebuild_index()


def save_database():
    """Persist the embedding database to disk as a pickle file."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(st.session_state.database, f)


cap = st.session_state.cap
capture_clicked = False


# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.title("Face Recognition")

    if st.session_state.mode == "recognition":
        st.success(f"Recognition mode \n {len(st.session_state.database)} people registered")

        # --- Enrollment ---
        name_input  = st.text_input("Name / Student ID")
        name_exists = name_input.strip() in st.session_state.database

        if name_exists:
            st.warning(f"**{name_input.strip()}** already registered.")
            start = st.button("Re-enroll",width='stretch')
        else:
            start = st.button("Start Enrollment",width='stretch')

        if start and name_input.strip():
            st.session_state.mode = "enrollment"
            st.session_state.enroll_name = name_input.strip()
            st.session_state.enroll_step = 0
            st.session_state.enroll_embeddings = []
            st.rerun()

        # --- Database management ---
        st.divider()
        st.subheader("Database")

        if st.session_state.database:
            with st.expander(f"Registered ({len(st.session_state.database)})"):
                for n in st.session_state.database:
                    st.write(f"• {n}  ({len(st.session_state.database[n])} imgs)")

        d1, d2 = st.columns(2)
        if d1.button("Reset DB",width='stretch'):
            st.session_state.database = {}
            save_database()
            rebuild_index()
            st.rerun()

        if d2.button("Retrain",width='stretch'):
            st.session_state.mode = "retraining"
            st.rerun()

    elif st.session_state.mode == "enrollment":
        st.warning(f"Enrolling: **{st.session_state.enroll_name}**")
        st.progress(st.session_state.enroll_step / 5)
        if st.session_state.enroll_step < 5:
            st.info(ANGLE_PROMPTS[st.session_state.enroll_step])

        c1, c2 = st.columns(2)
        capture_clicked = c1.button("📸 Capture",width='stretch')
        if c2.button("Cancel", width='stretch'):
            st.session_state.mode = "recognition"
            st.rerun()

    elif st.session_state.mode == "retraining":
        st.info("Retraining from dataset...")


# ───────────────────────── Retrain logic ───────────────────────
if st.session_state.mode == "retraining":
    if not os.path.isdir(DATASET_DIR):
        st.error(f"Dataset folder not found: `{DATASET_DIR}`")
        st.session_state.mode = "recognition"
        st.rerun()

    new_db   = {}
    progress = st.progress(0.0)
    status   = st.empty()
    people   = [d for d in sorted(os.listdir(DATASET_DIR))
                if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for pi, person_name in enumerate(people):
        person_dir = os.path.join(DATASET_DIR, person_name)
        status.write(f"Processing **{person_name}** ...")
        for fname in os.listdir(person_dir):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            img = preprocess(img)
            faces_found = model.get(img)
            if faces_found:
                new_db.setdefault(person_name, []).append(faces_found[0].embedding)
        progress.progress((pi + 1) / len(people))

    st.session_state.database = new_db
    save_database()
    rebuild_index()

    status.success(
        f"Done – {len(new_db)} people, "
        f"{sum(len(v) for v in new_db.values())} embeddings."
    )
    time.sleep(1.5)
    st.session_state.mode = "recognition"
    st.rerun()


# ───────────────────────── Main video loop ─────────────────────
frame_placeholder = st.empty()
msg_placeholder   = st.empty()

if not cap.isOpened():
    st.error("Cannot open camera.")
    st.stop()

frame_count = 0
faces       = []
do_capture  = capture_clicked

while True:
    ret, frame = cap.read()
    if not ret:
        msg_placeholder.error("Camera signal lost.")
        break

    frame   = cv2.flip(frame, 1)
    small   = cv2.resize(frame, (640, 480))
    display = small.copy()

    frame_count += 1
    if frame_count % DETECT_EVERY_N == 0:
        processed = preprocess(small)
        faces = model.get(processed)

    if st.session_state.mode == "recognition":
        db_mat = st.session_state.db_matrix
        db_nm  = st.session_state.db_names
        for face in faces:
            bbox = face.bbox.astype(int)
            emb  = face.embedding
            best_name, best_score = "Unknown", 0.0

            if db_mat is not None:
                norm = np.linalg.norm(emb)
                query = (emb / norm).astype(np.float32) if norm > 0 else emb
                scores = db_mat @ query
                idx = int(np.argmax(scores))
                best_score = float(scores[idx])
                best_name  = db_nm[idx]

            if best_score >= THRESHOLD:
                label, color = f"{best_name} ({best_score:.2f})", (0, 255, 0)
            else:
                label, color = f"Unknown ({best_score:.2f})", (0, 0, 255)

            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(display, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    elif st.session_state.mode == "enrollment":
        if faces:
            areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
            sel = int(np.argmax(areas))
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                if i == sel:
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (0, 215, 255), 3)
                else:
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (100, 100, 100), 1)

        if do_capture:
            do_capture = False
            capture_processed = preprocess(small)
            faces_now = model.get(capture_processed)
            if not faces_now:
                msg_placeholder.warning("No face detected – try again!")
            else:
                areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces_now]
                face  = faces_now[int(np.argmax(areas))]
                st.session_state.enroll_embeddings.append(face.embedding)

                person_dir = os.path.join(DATASET_DIR, st.session_state.enroll_name)
                os.makedirs(person_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(person_dir, f"angle_{st.session_state.enroll_step}.jpg"),
                    frame,
                )

                st.session_state.enroll_step += 1

                if st.session_state.enroll_step >= 5:
                    st.session_state.database[st.session_state.enroll_name] = (
                        st.session_state.enroll_embeddings
                    )
                    save_database()
                    rebuild_index()
                    st.session_state.mode = "recognition"
                    st.session_state.enroll_name = ""
                    st.session_state.enroll_embeddings = []
                    st.session_state.enroll_step = 0

                st.rerun()

    _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_placeholder.image(jpg.tobytes(), width='stretch')
