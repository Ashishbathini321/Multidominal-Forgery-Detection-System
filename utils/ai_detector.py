from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

labels = ["a real photograph", "an AI generated image"]

def is_ai_image(img):
    image = Image.fromarray(img[:, :, ::-1])

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    prob_ai = float(probs[1])

    return prob_ai >= 0.60, prob_ai