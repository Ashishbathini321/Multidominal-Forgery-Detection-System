import torch
import torchvision.transforms as transforms
from PIL import Image

# Load pretrained model
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_ai(image):
    img = Image.fromarray(image[:,:,::-1])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)

    prob = torch.softmax(out, dim=1)[0]

    # Fake logic (demo purpose)
    ai_score = float(prob[0])

    return ai_score > 0.6