from torchvision import transforms
from PIL import Image
import torch
import os

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 14 * 14, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class_names = ['airplane', 'apple', 'bird', 'cat', 'fish', 'flower', 'house', 'spider']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("./assets/model/custom_cnn_earlystop.pth", map_location=device))
model.to(device)
model.eval()

def cnn_predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        # print(f"Image: {os.path.basename(image_path)}")
        # print(f"Predicted class: {class_names[pred_class]} ({confidence:.2%} confidence)\n")
        return class_names[pred_class], confidence

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


if __name__ == "__main__":
    test_dir = 'test'
    for filename in sorted(os.listdir(test_dir)):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(test_dir, filename)
            
            guess, conf = cnn_predict_image(image_path)
            print(f"\nGuess: {guess}, Confidence: {conf:.2%}\n")
