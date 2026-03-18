import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2, json, pickle, time
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE, "db")
FACE_DB = os.path.join(DB_DIR, "face_db.pkl")
STUDENTS_DB = os.path.join(DB_DIR, "students.json")
ATTEND_DB = os.path.join(DB_DIR, "attendance.json")
DATASET = os.path.join(BASE, "dataset")
THRESHOLD = 0.5
DETECT_EVERY = 3
TOTAL_SESSIONS = 20
MAX_ABSENCES = 4
CLASS_START = 8
LATE_AFTER = 15
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PROMPTS = [
    "Step 1/5: Look directly at the camera",
    "Step 2/5: Turn your face to the left",
    "Step 3/5: Turn your face to the right",
    "Step 4/5: Tilt your head upward",
    "Step 5/5: Tilt your head downward",
]

def preprocess(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def biggest(faces):
    return int(np.argmax([(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]))

def read_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except: return default

def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)

st.set_page_config(page_title="CPV301 Attendance", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
*:not([class*="material-symbols"]) { font-family: 'JetBrains Mono', monospace !important; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import onnxruntime; onnxruntime.set_default_logger_severity(3)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        m = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        m.prepare(ctx_id=0, det_size=(320, 320))
    return m

model = load_model()

# session state init
if "cap" not in st.session_state:
    c = cv2.VideoCapture(0); c.set(cv2.CAP_PROP_BUFFERSIZE, 1); st.session_state.cap = c
if "db" not in st.session_state:
    try:
        with open(FACE_DB, "rb") as f: st.session_state.db = pickle.load(f)
    except: st.session_state.db = {}
if "students" not in st.session_state:
    st.session_state.students = read_json(STUDENTS_DB, {})
if "att" not in st.session_state:
    st.session_state.att = read_json(ATTEND_DB, {"sessions": [], "records": {}})
if "mode" not in st.session_state:
    st.session_state.mode = "recognition"
    st.session_state.enroll_data = {}
    st.session_state.enroll_step = 0
    st.session_state.enroll_embs = []
    st.session_state.show_clahe = False
    st.session_state.logged_today = set()

def rebuild():
    names, vecs = [], []
    for sid, embs in st.session_state.db.items():
        for v in embs:
            names.append(sid)
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
    st.session_state.mat = np.stack(vecs).astype(np.float32) if vecs else None
    st.session_state.names = names

if "mat" not in st.session_state: rebuild()

def save_db():
    os.makedirs(DB_DIR, exist_ok=True)
    with open(FACE_DB, "wb") as f: pickle.dump(st.session_state.db, f)

def log_att(sid):
    now = datetime.now()
    today, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    att = st.session_state.att
    if today not in att["sessions"]: att["sessions"].append(today)
    if sid not in att["records"]: att["records"][sid] = {}
    if today not in att["records"][sid]:
        att["records"][sid][today] = t
        write_json(ATTEND_DB, att)
    st.session_state.logged_today.add(sid)

def is_late(t):
    h, m, _ = map(int, t.split(":"))
    return h * 60 + m > CLASS_START * 60 + LATE_AFTER

def status(sid):
    att = st.session_state.att
    held = len(att["sessions"])
    recs = att["records"].get(sid, {})
    missed = held - len(recs)
    late = sum(1 for t in recs.values() if is_late(t))
    return missed > MAX_ABSENCES, len(recs), held, late

cap = st.session_state.cap
capture = False

# sidebar
with st.sidebar:
    st.title("CPV301 Attendance")
    
    if "cam_source" not in st.session_state: st.session_state.cam_source = "Laptop Webcam"
    cam_choice = st.radio("Camera Source", ["Laptop Webcam", "RTSP Camera"], 
                          index=0 if st.session_state.cam_source == "Laptop Webcam" else 1)
    if cam_choice != st.session_state.cam_source:
        st.session_state.cam_source = cam_choice
        if "cap" in st.session_state: st.session_state.cap.release()
        src = 0
        if cam_choice == "RTSP Camera":
            try:
                with open(os.path.join(BASE, r"C:\1MATERIAL\3LHMT\FPT\CPV301\assignment\url"), "r") as f:
                    content = f.read().strip()
                    src = content.split("=")[1].strip() if "=" in content else content
            except: pass
        c = cv2.VideoCapture(src)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state.cap = c
        st.rerun()

    held = len(st.session_state.att["sessions"])
    st.caption(f"Session {held} / {TOTAL_SESSIONS}")

    if st.session_state.mode == "recognition":
        st.success(f"Recognition - {len(st.session_state.db)} students")
        if st.button("View Dashboard", width='stretch'):
            st.session_state.mode = "dashboard"; st.rerun()

        st.divider()
        st.subheader("Enroll Student")
        name = st.text_input("Full Name *")
        sid = st.text_input("Student ID *")
        email = st.text_input("Email (optional)")

        exists = sid.strip() in st.session_state.db
        if exists: st.warning(f"{sid.strip()} already registered.")
        btn = st.button("Re-enroll" if exists else "Start Enrollment", width='stretch')

        if btn:
            if not name.strip() or not sid.strip():
                st.error("Full Name and Student ID are required.")
            else:
                st.session_state.mode = "enrollment"
                st.session_state.enroll_data = {"name": name.strip(), "student_id": sid.strip(), "email": email.strip(), "class": "CPV301"}
                st.session_state.enroll_step = 0
                st.session_state.enroll_embs = []
                st.rerun()

        st.divider()
        st.subheader("Database")
        if st.session_state.db:
            with st.expander(f"Registered ({len(st.session_state.db)})"):
                for s in st.session_state.db:
                    info = st.session_state.students.get(s, {})
                    st.write(f"- {s}: {info.get('name', 'N/A')}")
        d1, d2 = st.columns(2)
        if d1.button("Reset All", width='stretch'):
            st.session_state.db = {}
            st.session_state.students = {}
            st.session_state.att = {"sessions": [], "records": {}}
            save_db(); write_json(STUDENTS_DB, {}); write_json(ATTEND_DB, st.session_state.att)
            rebuild(); st.rerun()
        if d2.button("Retrain", width='stretch'):
            st.session_state.mode = "retraining"; st.rerun()

        st.divider()
        st.session_state.show_clahe = st.checkbox("Show CLAHE Preprocessing")

    elif st.session_state.mode == "enrollment":
        d = st.session_state.enroll_data
        st.info(f"Enrolling: {d['student_id']}")
        st.write(f"Name: {d['name']}")
        st.progress(st.session_state.enroll_step / 5)
        if st.session_state.enroll_step < 5: st.write(PROMPTS[st.session_state.enroll_step])
        c1, c2 = st.columns(2)
        capture = c1.button("Capture", width='stretch')
        if c2.button("Cancel", width='stretch'):
            st.session_state.mode = "recognition"; st.rerun()

    elif st.session_state.mode == "dashboard":
        st.info("Attendance Dashboard")
        if st.button("Back", width='stretch'):
            st.session_state.mode = "recognition"; st.rerun()

    elif st.session_state.mode == "retraining":
        st.info("Retraining...")

# dashboard
if st.session_state.mode == "dashboard":
    import pandas as pd
    st.header("CPV301 - Attendance Dashboard")
    att = st.session_state.att
    h = len(att["sessions"])
    st.write(f"Sessions held: {h} / {TOTAL_SESSIONS} | Max absences: {MAX_ABSENCES}")
    if st.session_state.students:
        rows = []
        for sid, info in st.session_state.students.items():
            recs = att["records"].get(sid, {})
            a, m = len(recs), h - len(recs)
            late = sum(1 for t in recs.values() if is_late(t))
            rows.append({"Student ID": sid, "Name": info.get("name",""), "Attended": a,
                         "Late": late, "Missed": m, "Rate": f"{a/h*100:.0f}%" if h else "0%",
                         "Status": "FAILED" if m > MAX_ABSENCES else "PASSED"})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    else: st.write("No students registered yet.")
    st.stop()

# retrain
if st.session_state.mode == "retraining":
    if not os.path.isdir(DATASET):
        st.error("Dataset folder not found"); st.session_state.mode = "recognition"; st.rerun()
    new = {}
    bar, txt = st.progress(0.0), st.empty()
    people = [d for d in sorted(os.listdir(DATASET)) if os.path.isdir(os.path.join(DATASET, d))]
    for i, p in enumerate(people):
        txt.write(f"Processing {p} ...")
        for f in os.listdir(os.path.join(DATASET, p)):
            if os.path.splitext(f)[1].lower() not in EXTS: continue
            img = cv2.imread(os.path.join(DATASET, p, f))
            if img is None: continue
            faces = model.get(preprocess(img))
            if faces: new.setdefault(p, []).append(faces[0].embedding)
        bar.progress((i+1)/len(people))
    st.session_state.db = new; save_db(); rebuild()
    txt.success(f"Done - {len(new)} students"); time.sleep(1.5)
    st.session_state.mode = "recognition"; st.rerun()

# camera layout
if st.session_state.show_clahe:
    c1, c2 = st.columns(2)
    c1.write("Original"); c2.write("After CLAHE")
    frame_slot, clahe_slot = c1.empty(), c2.empty()
else:
    frame_slot, clahe_slot = st.empty(), None

msg = st.empty()
if not cap.isOpened(): st.error("Cannot open camera."); st.stop()

fc, faces, do_cap = 0, [], capture
while True:
    ret, frame = cap.read()
    if not ret: msg.error("Camera signal lost."); break
    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (640, 480))
    display = small.copy()
    proc = preprocess(small)
    fc += 1
    if fc % DETECT_EVERY == 0: faces = model.get(proc)

    if st.session_state.mode == "recognition":
        mat, nms = st.session_state.mat, st.session_state.names
        for face in faces:
            bb = face.bbox.astype(int)
            emb = face.embedding
            bid, bscore = "Unknown", 0.0
            if mat is not None:
                n = np.linalg.norm(emb)
                sc = mat @ ((emb/n).astype(np.float32) if n > 0 else emb)
                idx = int(np.argmax(sc))
                bscore, bid = float(sc[idx]), nms[idx]
            if bscore >= THRESHOLD:
                info = st.session_state.students.get(bid, {})
                nm = info.get("name", bid)
                if bid not in st.session_state.logged_today: log_att(bid)
                failed, _, _, _ = status(bid)
                label = f"FAILED - {nm} ({bscore:.2f})" if failed else f"{nm} ({bscore:.2f})"
                color = (0, 0, 255) if failed else (0, 255, 0)
            else:
                label, color = f"Unknown ({bscore:.2f})", (0, 0, 255)
            cv2.rectangle(display, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
            cv2.putText(display, label, (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    elif st.session_state.mode == "enrollment":
        if faces:
            sel = biggest(faces)
            for i, face in enumerate(faces):
                bb = face.bbox.astype(int)
                cl = (0,215,255) if i == sel else (100,100,100)
                cv2.rectangle(display, (bb[0],bb[1]), (bb[2],bb[3]), cl, 3 if i==sel else 1)
        if do_cap:
            do_cap = False
            fn = model.get(proc)
            if not fn: msg.warning("No face detected - try again!")
            else:
                face = fn[biggest(fn)]
                st.session_state.enroll_embs.append(face.embedding)
                sid = st.session_state.enroll_data["student_id"]
                pd = os.path.join(DATASET, sid); os.makedirs(pd, exist_ok=True)
                cv2.imwrite(os.path.join(pd, f"angle_{st.session_state.enroll_step}.jpg"), frame)
                st.session_state.enroll_step += 1
                if st.session_state.enroll_step >= 5:
                    st.session_state.db[sid] = st.session_state.enroll_embs
                    save_db(); rebuild()
                    st.session_state.students[sid] = st.session_state.enroll_data
                    write_json(STUDENTS_DB, st.session_state.students)
                    st.session_state.mode = "recognition"
                    st.session_state.enroll_data = {}
                    st.session_state.enroll_embs = []
                    st.session_state.enroll_step = 0
                st.rerun()

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(display, ts, (10, display.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_slot.image(jpg.tobytes(), width='stretch')
    if clahe_slot:
        _, j2 = cv2.imencode(".jpg", proc, [cv2.IMWRITE_JPEG_QUALITY, 80])
        clahe_slot.image(j2.tobytes(), width='stretch')
