# Human Activity Recognition (CNN-LSTM)

A video-based deep learning system that classifies human activities from short video clips using a hybrid **CNN + LSTM** architecture. The CNN extracts spatial features from individual frames, while the LSTM models temporal dependencies across the frame sequence — enabling the model to distinguish activities that look visually similar in a single frame but differ in motion pattern (e.g., *Walking* vs. *Walking While Using Phone*).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![Status](https://img.shields.io/badge/Test%20Accuracy-96.9%25-brightgreen)

## Activities Classified (7 classes)
- Clapping
- Meet and Split
- Sitting
- Standing Still
- Walking
- Walking While Reading Book
- Walking While Using Phone

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **96.86%** |
| Test Loss | 0.1325 |
| Macro Avg F1 | 0.97 |
| Weighted Avg F1 | 0.97 |
| Training Epochs | 20 |
| Final Train Accuracy | 99.45% |
| Final Validation Accuracy | 96.63% |
| Dataset Size | 1,113 videos across 7 classes |
| Test Set Size | 223 videos |

**Per-class performance** (precision / recall / F1) is strong across all classes, with *Walking While Using Phone* achieving perfect (1.00) precision, recall, and F1 — the full classification report is generated in the notebook via `sklearn.metrics.classification_report`.

## Tech Stack
- **OpenCV (`cv2`)** — video I/O and frame processing
- **TensorFlow / Keras** — CNN + LSTM model definition, training, inference
- **NumPy** — array/tensor handling
- **scikit-learn** — label encoding, train/test split, classification metrics
- **Matplotlib** — training curves and frame visualization
- **tqdm** — progress tracking during preprocessing

## Project Structure
```
Activity_Recognition/
├── Dataset/                             # Human Activity Recognition video dataset (class-labeled folders)
├── Activity Recognition Program.ipynb   # Main notebook: pipeline + model + training + inference
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Excludes venv, dataset, model weights, etc.
└── README.md
```

## Pipeline

### 1. Data Ingestion
Videos are organized in class-labeled folders (one folder per activity, 7 total). Each folder is scanned and every video file is processed and tagged with its class label. **1,113 videos** processed across the 7 classes.

### 2. Video Preprocessing (`preprocess_video`)
For each video:
- Skip videos with fewer than 100 total frames (quality/length filter)
- Sample every **4th frame**, capped at **25 frames** per video
- Convert each sampled frame: BGR → **grayscale**
- Resize to **24×24**
- Normalize pixel values to `[0, 1]`

Output: a `(25, 24, 24)` array per video — final dataset shape: **`(1113, 25, 24, 24)`**.

### 3. Dataset Assembly
- Processed videos and labels collected into `Videos` and `Activity` arrays
- Labels encoded: `LabelEncoder` → integer labels → `to_categorical` → one-hot vectors
- Train/test split: 80/20 (`random_state=42`)

### 4. Model Architecture
```
Input: (25 timesteps, 24, 24, 1)
  → TimeDistributed(Conv2D 32, 3x3, ReLU) → TimeDistributed(MaxPool 2x2) → Dropout(0.3)
  → TimeDistributed(Conv2D 64, 3x3, ReLU) → TimeDistributed(MaxPool 2x2) → Dropout(0.3)
  → TimeDistributed(Flatten)
  → LSTM(128)
  → Dropout(0.5)
  → Dense(7, softmax)

Optimizer: Adam | Loss: Categorical Crossentropy | Metric: Accuracy
```
The `TimeDistributed` wrapper applies the same CNN independently to each of the 25 frames; the LSTM then consumes the resulting sequence of per-frame feature vectors to learn temporal motion patterns.

### 5. Training & Evaluation
- 20 epochs, batch size 8, 20% validation split
- Training accuracy climbed from 16.6% (epoch 1) to **99.45%** (epoch 20); validation accuracy reached **96.63%**
- Model checkpoint saved as `activity_recognition_model.h5`
- Evaluated on held-out test set: **96.86% accuracy**, 0.1325 loss
- Training/validation accuracy and loss curves plotted (see notebook output)
- Per-class precision/recall/F1 via `classification_report` — macro and weighted averages both 0.97

### 6. Inference
`predict_activity(video_path)` runs the same preprocessing on a new video, reshapes it to match the model's expected input, and returns the predicted activity label (decoded via the fitted `LabelEncoder`).

## Setup

```bash
git clone <your-repo-url>
cd Activity_Recognition
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Usage

1. Organize your dataset as:
   ```
   Dataset/
   ├── Walking/
   │   ├── Walking (1).mp4
   │   └── ...
   ├── Clapping/
   │   └── ...
   └── ... (7 class folders total)
   ```
2. Update the `home` path in the notebook to point to your dataset directory.
3. Run all cells to preprocess data, train the model, and evaluate it.
4. Run inference on a new video:
   ```python
   predict_activity("path/to/video.mp4")
   ```

## Known Issues / TODO
- [ ] **Bug**: `predict_activity` ignores its `path` argument and always evaluates the hardcoded `pth` variable inside the function — needs to use the passed-in `path` parameter instead.
- [ ] Inference reshape (`(1, 25, 24, 24)`) drops the channel dimension used during training (`(25, 24, 24, 1)`) — works currently but is structurally inconsistent with the trained input shape; should be `(1, 25, 24, 24, 1)` for robustness.
- [ ] Dataset paths are hardcoded as absolute Windows paths — replace with relative/config-driven paths for portability.
- [ ] `tensorflow_hub` is imported but not used in the current pipeline — remove if not needed, or document intended use (e.g., a pretrained frame encoder).
- [ ] Model saved in legacy HDF5 (`.h5`) format — TensorFlow recommends migrating to the native `.keras` format.
- [ ] No fixed random seed for frame sampling — reproducibility relies only on `train_test_split`'s `random_state`.

## Possible Extensions
- Swap the custom CNN encoder for a pretrained backbone (e.g., MobileNet via `tensorflow_hub`) for stronger spatial features
- Increase frame resolution/count if compute allows, and benchmark the accuracy trade-off
- Add a real-time webcam inference loop
- Convert model to TFLite/ONNX for lightweight edge deployment
- Add confusion matrix visualization alongside the classification report

## License
Add a license of your choice (MIT is common for portfolio projects).