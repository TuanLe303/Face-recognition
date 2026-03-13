"""
create_db.py – Build the face-embedding database from a folder of images.

Pipeline (per image):
    1. Read image with OpenCV
    2. Pre-process  (Gaussian blur denoise + brightness normalization)
    3. Detect face  (RetinaFace via InsightFace buffalo_l)
    4. Extract 512-d embedding (ArcFace)
    5. Store embedding in a dict keyed by person name
    6. Serialize to a .pkl file

Dataset layout:
    dataset/
    ├── PersonA/
    │   ├── img1.jpg
    │   └── img2.png
    └── PersonB/
        └── img1.jpg

Usage:
    python code/create_db.py                         # defaults
    python code/create_db.py --dataset path/to/imgs  # custom dataset
    python code/create_db.py --db path/to/db.pkl     # custom output
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ─────────── Defaults ───────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset")
DEFAULT_DB      = os.path.join(BASE_DIR, "db", "face_db.pkl")
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────── Pre-processing ────────────────────────────────────
def preprocess(img: np.ndarray) -> np.ndarray:
    """Light pre-processing applied before face detection.
    - Gaussian blur (3×3) to reduce camera/compression noise.
    - Min-max normalization to balance brightness/contrast."""
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img


# ─────────── Main logic ────────────────────────────────────────
def build_database(dataset_dir: str, db_path: str) -> None:
    """Scan dataset_dir for person sub-folders, extract face embeddings,
    and save the resulting database to db_path."""

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        print("        Create it and add sub-folders for each person.")
        return

    people = [d for d in sorted(os.listdir(dataset_dir))
              if os.path.isdir(os.path.join(dataset_dir, d))]

    if not people:
        print(f"[ERROR] No person folders found in: {dataset_dir}")
        return

    # Initialize InsightFace model
    print("[INFO] Loading InsightFace model (buffalo_l) ...")
    model = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.prepare(ctx_id=0, det_size=(640, 640))

    face_db: dict[str, list[np.ndarray]] = {}
    total_images = 0
    total_faces  = 0

    for person_name in people:
        person_dir = os.path.join(dataset_dir, person_name)
        image_files = [f for f in sorted(os.listdir(person_dir))
                       if os.path.splitext(f)[1].lower() in IMG_EXTS]

        if not image_files:
            print(f"\n[{person_name}]  (no images found, skipping)")
            continue

        print(f"\n[{person_name}]")
        for fname in image_files:
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] Could not read: {fname}")
                continue

            total_images += 1

            # Pre-process → detect → extract embedding
            img   = preprocess(img)
            faces = model.get(img)

            if not faces:
                print(f"  [WARN] No face detected: {fname}")
                continue

            embedding = faces[0].embedding
            face_db.setdefault(person_name, []).append(embedding)
            total_faces += 1
            print(f"  [OK]   {fname}")

    # Save database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "wb") as f:
        pickle.dump(face_db, f)

    # Summary
    print("\n" + "=" * 50)
    print(f"  Database saved to : {db_path}")
    print(f"  People            : {len(face_db)}")
    print(f"  Total images      : {total_images}")
    print(f"  Total embeddings  : {total_faces}")
    print("=" * 50)
    for name, vecs in face_db.items():
        print(f"  {name}: {len(vecs)} embeddings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build face-embedding database from a dataset folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Path to the dataset folder (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Path to save the .pkl database (default: {DEFAULT_DB})",
    )
    args = parser.parse_args()

    build_database(args.dataset, args.db)
