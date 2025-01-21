from torchvision import transforms
from PIL import Image

def predict_image(model, img_path, idx_to_class):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),          
    ])

    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)

    model.eval()
    output = model(image)
    _, pred = output.max(dim=1)

    return idx_to_class[pred.item()], pred.item()