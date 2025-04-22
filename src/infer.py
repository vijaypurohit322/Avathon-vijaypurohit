import torch
from models.cnn_model import CaptchaCNN
from config import MODEL_PATH, SEQ_LENGTH
from utils import preprocess_image, transform

CHARS = "0123456789"

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image = transform(preprocess_image(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=2).squeeze()
        result = ''.join([CHARS[i] for i in pred])
    return result

if __name__ == '__main__':
    test_img_path = "data/validation-images/image_val_10.png"
    print("Predicted:", predict(test_img_path))
