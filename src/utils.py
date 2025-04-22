import cv2
from PIL import Image
from torchvision import transforms

from config import IMAGE_WIDTH, IMAGE_HEIGHT

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.medianBlur(img, 3)
    return Image.fromarray(img)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
