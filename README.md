# Human Activity Recognition (CNN + LSTM)

A deep learning pipeline that classifies human activities from short video clips using a hybrid **CNN + LSTM** architecture. The CNN extracts spatial features from individual frames, while the LSTM models temporal dependencies across the frame sequence — enabling the model to distinguish activities that look similar in a single frame but differ in motion pattern (e.g., *Walking* vs. *Walking While Using Phone*).

## Activities Classified
- Walking
- Clapping
- Reading While Walking
- Walking While Using Phone
- *(and other classes present in the dataset directory — `num_classes = 7`)*

## Tech Stack
- **OpenCV (`cv2`)** — video I/O and frame processing
- **TensorFlow / Keras** — CNN + LSTM model definition, training, inference
- **NumPy** — array/tensor handling
- **scikit-learn** — label encoding, train/test split, classification metrics
- **Matplotlib** — training curves and frame visualization
- **tqdm** — progress tracking during preprocessing

## Pipeline

### 1. Data Ingestion
Videos are organized in class-labeled folders (one folder per activity). Each folder is scanned, and every video file within it is processed and tagged with its class label.

### 2. Video Preprocessing (`preprocess_video`)
For each video:
- Skip videos with fewer than 100 total frames (quality/length filter)
- Sample every **4th frame**, capped at **25 frames** per video
- Convert each sampled frame: BGR → **grayscale**
- Resize to **24×24**
- Normalize pixel values to `[0, 1]`

Output: a `(25, 24, 24)` array per video capturing a compact spatiotemporal fingerprint of the activity.

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
  → Dense(num_classes, softmax)

Optimizer: Adam | Loss: Categorical Crossentropy | Metric: Accuracy
```
The `TimeDistributed` wrapper applies the same CNN independently to each of the 25 frames; the LSTM then consumes the resulting sequence of per-frame feature vectors to learn temporal patterns.

### 5. Training & Evaluation
- 20 epochs, batch size 8, 20% validation split
- Model checkpoint saved as `activity_recognition_model.h5`
- Evaluated on held-out test set (loss + accuracy)
- Training/validation accuracy and loss curves plotted
- Per-class precision/recall/F1 via `classification_report`

### 6. Inference
`predict_activity(video_path)` runs the same preprocessing on a new video, reshapes it to match the model's expected input, and returns the predicted activity label (decoded via the fitted `LabelEncoder`).

## Setup

```bash
pip install opencv-python numpy tensorflow tensorflow-hub scikit-learn matplotlib tqdm
```

## Usage

1. Organize your dataset as:
   ```
   Human Activity Recognition - Video Dataset/
   ├── Walking/
   │   ├── Walking (1).mp4
   │   └── ...
   ├── Clapping/
   │   └── ...
   └── ...
   ```
2. Update the `home` path in the notebook to point to your dataset directory.
3. Run all cells to preprocess data, train the model, and evaluate it.
4. Run inference on a new video:
   ```python
   predict_activity("path/to/video.mp4")
   ```

## Known Issues / TODO
- [ ] **Bug**: `predict_activity` ignores its `path` argument and always evaluates the hardcoded `pth` variable — needs to use the passed-in `path` parameter.
- [ ] **Shape mismatch risk**: inference reshapes to `(1, 25, 24, 24)`, missing the channel dimension used in training (`(1, 25, 24, 24, 1)`).
- [ ] Dataset paths are hardcoded as absolute Windows paths — replace with relative/config-driven paths for portability.
- [ ] `tensorflow_hub` is imported but not used in the current pipeline — remove if not needed, or document its intended use (e.g., a pretrained frame encoder).
- [ ] No fixed random seed for frame sampling — training/eval reproducibility relies only on `train_test_split`'s `random_state`.
- [ ] Consider data augmentation (frame jitter, horizontal flip) to improve generalization given the low resolution (24×24).

## Possible Extensions
- Swap the custom CNN encoder for a pretrained backbone (e.g., MobileNet) via `tensorflow_hub` for stronger spatial features
- Increase frame resolution/count if compute allows, and benchmark accuracy trade-off
- Add real-time webcam inference loop
- Convert model to TFLite/ONNX for lightweight deployment
