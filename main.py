from flask import Flask, request, send_file
from PIL import Image
import torch
from torchvision import transforms
import os

app = Flask(__name__)

# DeepLabv3 Model
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/segment-cloth', methods=['POST'])
def segment_cloth():
    if 'file' not in request.files:
        return {"error": "No file part in the request"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No file selected for uploading"}, 400

    input_path = "input.png"
    output_path = "output.png"
    file.save(input_path)

    try:
        image = Image.open(input_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0) 

        with torch.no_grad():
            output = model(input_tensor)['out'][0] 

        output_predictions = torch.argmax(output, dim=0).byte().cpu().numpy()

        segmented_image = Image.fromarray(output_predictions * 255)  
        segmented_image.save(output_path)

    except Exception as e:
        return {"error": str(e)}, 500

    return send_file(output_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
