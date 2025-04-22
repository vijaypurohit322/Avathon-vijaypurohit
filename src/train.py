import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CaptchaDataset
from models.cnn_model import CaptchaCNN
from config import *


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = CaptchaDataset(CSV_PATH, 'train-images')
    val_ds = CaptchaDataset(CSV_PATH, 'validation-images')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CaptchaCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = sum(criterion(preds[:, i, :], labels[:, i]) for i in range(SEQ_LENGTH))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct_full, correct_chars, total_chars = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                pred_labels = preds.argmax(dim=2)

                correct_full += (pred_labels == labels).all(dim=1).sum().item()
                correct_chars += (pred_labels == labels).sum().item()
                total_chars += labels.numel()

        print(f"Exact Match Accuracy: {correct_full / len(val_ds):.4f}")
        print(f"Per-Character Accuracy: {correct_chars / total_chars:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    train_model()
