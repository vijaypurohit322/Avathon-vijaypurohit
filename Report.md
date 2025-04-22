# CAPTCHA Recognition Report

## Objective
Build a robust model to extract 6-digit numeric strings from noisy CAPTCHA images.

## Approach
- Image preprocessing with OpenCV: thresholding, resizing, denoising
- CNN model to predict each digit in sequence
- Trained using cross-entropy per digit
- Evaluated using exact-match and character-wise accuracy

## Architecture
- 2 Conv layers → FC → output 6×10 logits
- No RNNs, no attention for simplicity and speed

## Results
| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 0.942 |
| Per-Character Accuracy | 0.981 |

## Challenges
- Noise and varied fonts in images
- Zeros at the start being dropped by label parser (solved by treating all labels as string)

## Future Work
- Add CRNN and CTC loss

## How to Run
See `README.md`

## Dependencies
See `requirements.txt`
