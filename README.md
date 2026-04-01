# YOLO Facial Emotion Recognition on macOS (M2)

A local, end-to-end facial emotion recognition pipeline optimized for Apple Silicon (M2), using YOLOv8 to detect and classify five key emotions in real-time.

## Goal & Motivation
This project demonstrates the feasibility of training and deploying a lightweight computer vision model locally on a consumer-grade macOS device. It solves the problem of needing external cloud resources for ML workloads by leveraging the **Metal Performance Shaders (MPS)** backend for hardware acceleration.

## Results & Metrics
The model was trained for 8 epochs with a batch size of 16 using the `yolov8n` (nano) architecture.

| Metric | Score |
| :--- | :--- |
| **mAP50 (B)** | **0.7099** |
| **mAP50-95 (B)** | **0.6408** |
| Precision (B) | 0.6191 |
| Recall (B) | 0.7425 |
| Training Time | ~7.8 hours (MacBook Air M2) |

*Evaluation performed on a held-out test split from the Kaggle Facial Emotion dataset.*

## Features
- **End-to-End Pipeline:** Single Jupyter Notebook covering data preparation, training, evaluation, and inference.
- **Multi-Source Inference:** Supports static images, stored videos, and real-time webcam feeds.
- **Emotion Classes:** Angry, Fear, Happy, Sad, Surprise.
- **Hardware Optimized:** Specifically configured for Apple Silicon (M2) using the `mps` device backend.

## Tech Stack
| Component | Technology |
| :--- | :--- |
| **Model Architecture** | YOLOv8 (nano) |
| **Framework** | Ultralytics YOLOv8, PyTorch |
| **Hardware Acceleration**| Metal Performance Shaders (MPS) |
| **Language** | Python 3.12 |
| **Environment** | Jupyter Notebook, macOS (M2) |
| **Libraries** | OpenCV, Pandas, Matplotlib |

## Key Engineering Challenges

### 1. Apple Silicon (MPS) Acceleration
Training YOLO models on macOS requires specific PyTorch configurations to utilize the M2's GPU via the Metal Performance Shaders (MPS) backend. I had to resolve initial dependency conflicts between `ultralytics` and `torch` versions to ensure the `device='mps'` flag was properly recognized and utilized, significantly reducing training time from days to hours.

### 2. Dataset Format Migration
The Kaggle emotion dataset was originally provided in a directory-per-class format (standard for classification), which is incompatible with YOLO's object detection format (images + `.txt` label files). I implemented a custom preprocessing script within the notebook to convert the raw classification dataset into a valid YOLO detection dataset, including bounding box generation and train/val/test split management.

### 3. Real-time Inference Performance
Balancing inference speed with detection accuracy on a webcam feed was challenging. I implemented a `frame_skip` mechanism and resolution downsampling (to 480p) to maintain a consistent 10+ FPS during real-time emotion detection, preventing lag while maintaining a high enough confidence threshold (`conf=0.4-0.6`) to avoid false positives.

## Project Structure
```text
.
├── emotions.ipynb       # Main project notebook (Training/Eval/Inference)
├── dataset.yaml         # YOLO dataset configuration
├── yolov8n.pt           # Pre-trained base weights
├── data/                # Processed dataset (images/labels)
│   ├── raw/             # Original Kaggle data
│   └── processed/       # YOLO-formatted data
├── models/              # Saved model weights
└── runs/                # Training logs and inference results
```

## Setup & Run
1. **Environment:** Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install ultralytics torch torchvision opencv-python pandas matplotlib
   ```
2. **Hardware Check:** Ensure you are on macOS 13+ with an M-series chip.
3. **Execution:** Open `emotions.ipynb` and execute cells sequentially.
4. **Webcam Inference:** Use the `inference_model.predict(source=0)` call in the final section to launch the live demo.

## Limitations & Known Issues
- **Limited Classes:** Only five emotions are currently supported (Angry, Fear, Happy, Sad, Surprise). Neutral and Disgust classes were omitted to improve class balance.
- **Nano Model Constraints:** While `yolov8n` is fast, it occasionally struggles with small faces or poor lighting compared to larger variants like `yolov8m`.
- **Typo in Data Directory:** One raw data folder is named `Suprise` (instead of `Surprise`), which is handled internally by the dataset loader but should be noted during manual inspection.

## What I Learned
- **MPS Backend:** How to effectively harness Apple Silicon's GPU for machine learning without external GPUs.
- **YOLO Ecosystem:** The nuances of the Ultralytics pipeline, from `.yaml` configuration to hyperparameter tuning.
- **Data Engineering:** The importance of robust data transformation scripts when adapting "off-the-shelf" datasets for specialized model architectures.
- **Inference Optimization:** Trade-offs between resolution, confidence thresholds, and frame-skipping for real-time performance.

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch MPS Backend Guide](https://pytorch.org/docs/stable/notes/mps.html)
- [Kaggle Facial Emotion Dataset](https://www.kaggle.com/)

## Author
[chsuiinx](https://github.com/chsuiinx)
