# Image properties
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60
SEQ_LENGTH = 6
NUM_CLASSES = 10  # Digits 0-9

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Paths
CSV_PATH = "data/captcha_data.csv"
TRAIN_DIR = "data/train-images"
VAL_DIR = "data/validation-images"
MODEL_PATH = "model_100_epoch.pth"
