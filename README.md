# CPV301 Face Recognition Attendance System

Real-time face recognition attendance system for class CPV301 using [InsightFace](https://github.com/deepinsight/insightface) (buffalo_l) with a Streamlit web UI.

## How It Works

```
Camera Frame
     |
     v
+---------------------------------------------+
|  Pre-processing                             |
|  Gaussian blur 5x5 -> noise reduction       |
|  CLAHE on LAB L-channel -> fix lighting     |
+---------------------+-----------------------+
                      v
+---------------------------------------------+
|  RetinaFace (det_10g)                       |
|  Detects faces -> bounding boxes +          |
|  5 keypoints (eyes, nose, mouth)            |
+---------------------+-----------------------+
                      v
+---------------------------------------------+
|  ArcFace (w600k_r50)                        |
|  Aligned face crop -> 512-dim embedding     |
|  Trained on 600K identities                 |
+---------------------+-----------------------+
                      v
+---------------------------------------------+
|  Matching                                   |
|  query (1x512) @ database (Nx512)^T         |
|  = cosine similarity scores                 |
|  max(score) >= 0.5 -> recognized            |
+---------------------------------------------+
                      v
+---------------------------------------------+
|  Attendance Logging                         |
|  Auto-logs student for today's session      |
|  >4 absences out of 20 sessions = FAILED    |
+---------------------------------------------+
```

Both models are bundled in InsightFace's **buffalo_l** pack and run via ONNX Runtime (GPU or CPU).

## Project Structure

```
Face-recognition/
├── code/
│   ├── app.py           # Streamlit attendance app
│   ├── create_db.py     # CLI: build face database from images
│   └── compare.py       # CLI: before/after CLAHE comparison
├── dataset/             # Face images organized by student ID
│   ├── HE200666/
│   │   ├── angle_0.jpg
│   │   └── angle_1.jpg
│   └── HE200777/
│       └── photo1.jpg
├── db/
│   ├── face_db.pkl      # Face embeddings (auto-generated)
│   ├── students.json    # Student info (auto-generated)
│   └── attendance.json  # Attendance records (auto-generated)
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/<your-username>/Face-recognition.git
cd Face-recognition

conda create -n facerec python=3.10 -y
conda activate facerec
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU: `pip install onnxruntime-gpu` (requires CUDA 12)

### 3. (Optional) Pre-build database from images

```bash
python code/create_db.py
python code/create_db.py --dataset path/to/images --db path/to/output.pkl
```

## Usage

```bash
streamlit run code/app.py
```

### Recognition Mode (default)

Camera detects faces and matches them against the database. Recognized students are automatically logged for today's session.

- Green box: `StudentID Name (attended/total)` - attendance logged
- Red box: `FAILED StudentID Name (attended/total)` - more than 4 absences

### Enrollment

1. Enter Full Name (required) and Student ID (required) in the sidebar
2. Click "Start Enrollment"
3. Follow the 5-angle prompts and click "Capture" for each
4. Student is added to the database

### Dashboard

Click "View Dashboard" to see all students with:
- Sessions attended / missed
- Attendance rate
- Status: PASSED or FAILED (>4 absences = FAILED)

### CLAHE Preprocessing

Check "Show CLAHE Preprocessing" in the sidebar to see original vs processed frames side by side.

### Compare Tool (standalone)

```bash
python code/compare.py
```

Shows before/after CLAHE in an OpenCV window. Press `q` to quit.

## Attendance Rules

- Class: CPV301
- Total sessions: 20
- Maximum allowed absences: 4 (20%)
- More than 4 absences = automatic FAILED status

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: onnxruntime` | `pip install onnxruntime` |
| `libiomp5md.dll` error | Handled automatically via `KMP_DUPLICATE_LIB_OK` |
| CUDA errors | Falls back to CPU automatically |
| Camera not opening | Close other apps using the webcam |

## Tech Stack

- **Detection + Recognition**: InsightFace buffalo_l (RetinaFace + ArcFace)
- **Inference**: ONNX Runtime (CPU / CUDA)
- **UI**: Streamlit with JetBrains Mono font
- **Pre-processing**: OpenCV (Gaussian blur + CLAHE)
- **Storage**: pickle (embeddings), JSON (students + attendance)
