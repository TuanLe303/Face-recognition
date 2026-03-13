# 🎓 Face Recognition System

A real-time face recognition system built with [InsightFace](https://github.com/deepinsight/insightface) (buffalo_l) for face detection & recognition, powered by a Streamlit web UI.

## Features

| Feature | Description |
|---------|-------------|
| **RTSP / Webcam streaming** | Supports both IP cameras (RTSP) and local webcams |
| **Face detection** | RetinaFace detector via InsightFace (`det_10g`) |
| **Face recognition** | ArcFace embeddings with cosine similarity matching |
| **Pre-processing** | Gaussian blur denoising + brightness/contrast normalization |
| **Live enrollment** | Register new faces from 5 angles directly through the UI |
| **Database management** | Reset, retrain, and view registered faces from the sidebar |

## How It Works

```
Camera Frame
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  1. Pre-processing                                       │
│     Gaussian blur (3×3) → denoise                        │
│     Min-max normalization → balance brightness/contrast  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  2. Face Detection — RetinaFace (det_10g)                │
│     Input: full frame → Output: bounding boxes +         │
│     5 facial keypoints (eyes, nose, mouth corners)       │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  3. Face Recognition — ArcFace (w600k_r50)               │
│     Input: aligned face crop → Output: 512-d embedding   │
│     (trained on 600K identities)                         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  4. Matching — Vectorized cosine similarity              │
│     query (1×512) @ database (N×512)ᵀ = scores (1×N)     │
│     If max(score) ≥ 0.5 → recognized                    │
└──────────────────────────────────────────────────────────┘
```

Both models are bundled in InsightFace's **buffalo_l** model pack and run via ONNX Runtime (GPU or CPU).

## Project Structure

```
Face-recognition/
├── code/
│   ├── app.py              # Streamlit web app (main application)
│   └── create_db.py        # CLI tool to build face database from images
├── dataset/                # Face images (organized by person)
│   └── <PersonName>/
│       ├── img1.jpg
│       └── img2.png
├── db/
│   └── face_db.pkl         # Face embedding database (auto-generated)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Face-recognition.git
cd Face-recognition
```

### 2. Create a Python environment

```bash
# Using conda
conda create -n face-recognition python=3.10 -y
conda activate face-recognition

# Or using venv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support (optional):** If you have an NVIDIA GPU with CUDA 12, install `onnxruntime-gpu` instead of `onnxruntime` for faster inference:
> ```bash
> pip install onnxruntime-gpu
> ```

### 4. Prepare the dataset

Add face images in the `dataset/` folder with the following structure:

```
dataset/
├── John_Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.png
├── Jane_Smith/
│   ├── img1.jpg
│   └── img2.jpg
└── ...
```

- Each sub-folder name = the person's name or student ID
- At least 1 image per person (more images = better accuracy)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 5. Build the face database

```bash
python code/create_db.py
```

Options:

```bash
python code/create_db.py --help
python code/create_db.py --dataset path/to/custom_dataset
python code/create_db.py --db path/to/output.pkl
```

## Usage

### Run the web app

```bash
streamlit run code/app.py
```

This opens a browser window with the live camera feed and sidebar controls.

### Modes

1. **Recognition mode** (default): Detects faces in the camera feed, matches against the database, and displays the name + confidence score.

2. **Enrollment mode**: Enter a name/student ID, then capture 5 photos at different angles (front, left, right, up, down). Images are saved to `dataset/` and embeddings to the database.

3. **Retrain**: Rebuilds the entire database from all images in `dataset/`. Useful after manually adding/removing images.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: onnxruntime` | Run `pip install onnxruntime` (or `onnxruntime-gpu` for GPU) |
| OpenMP `libiomp5md.dll` error | Already handled in code via `KMP_DUPLICATE_LIB_OK=TRUE` |
| CUDA provider errors | Install CUDA Toolkit 12 + cuDNN, or ignore (falls back to CPU) |
| Camera not opening | Check that no other app is using the webcam |
| Low recognition accuracy | Add more enrollment images, ensure good lighting |

## Tech Stack

- **Face Detection / Recognition**: [InsightFace](https://github.com/deepinsight/insightface) (buffalo_l – RetinaFace + ArcFace)
- **Inference**: [ONNX Runtime](https://onnxruntime.ai/) (CPU or CUDA)
- **Web UI**: [Streamlit](https://streamlit.io/)
- **Pre-processing**: OpenCV (Gaussian blur + brightness normalization)

## License

This project is for educational purposes (FPT University – CPV301).
