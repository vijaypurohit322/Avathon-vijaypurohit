# README
This project is an AI-based CAPTCHA recognition system that detects and extracts 6-digit numeric strings from noisy, distorted images using deep learning.

---

## Directory Structure
```
Avathon-vijaypurohit/
├── data/
│   ├── train-images/               # Training CAPTCHA images
│   ├── validation-images/          # Validation CAPTCHA images
│   └── captcha_data.csv            # CSV with image paths and labels
├── models/
│   └── cnn_model.py                # CNN model architecture
├── src/
│   ├── train.py                    # Training script
│   ├── infer.py                    # Inference script
│   ├── dataset.py                  # Dataset class
│   ├── utils.py                    # Preprocessing and transforms
│   └── config.py                   # Constants and hyperparameters
├── model.pth                       # Trained model weights (generated after training)
├── requirements.txt                # Dependencies
├── README.md                       # Project overview and instructions
└── Report.md                       # Technical summary and evaluation
```

---

## Getting Started

### 1. Clone the repository
```bash
git https://github.com/vijaypurohit322/Avathon-vijaypurohit.git
cd Avathon-vijaypurohit
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```

### 4. Run inference
```bash
python src/infer.py
```

> Make sure `captcha_data.csv`, `train-images/`, and `validation-images/` are inside the `data/` directory as per structure.

---

## Model Details

- Architecture: 2-layer CNN + Fully Connected for multi-digit classification
- Loss: Cross-entropy per digit
- Accuracy Metrics:
  - Close match (entire 6-digit string is correct in most of the cases)
  - Per-character accuracy (correct digits across normal positions)

---

## Sample Evaluation Output
```
Exact Match Accuracy: 0.9420
Per-Character Accuracy: 0.9812
```

---

## Future Enhancements
- Add elastic distortions, skewing, and background noise augmentation
- CRNN or Transformer-based sequence decoder
- CTC loss for better alignment handling
- Model quantization for deployment on edge devices

---

## Acknowledgments
Built with PyTorch, OpenCV, and love.

