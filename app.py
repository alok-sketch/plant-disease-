from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
import json
import cv2
import numpy as np
from plant_disease_model import PlantDiseaseModel
from language_utils import LanguageHandler
from api.soil_analysis import soil_bp

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

model = None
language_handler = None
class_names = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []
    
    def save_gradients(grad):
        gradients.append(grad)
    
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradients)
    
    target_layer = model.get_last_conv_layer()
    
    if target_layer is None:
        return np.zeros((224, 224))
    
    handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        output = model(image_tensor)
        pred_score = output.max()
        
        model.zero_grad()
        pred_score.backward()
        
        gradients = gradients[0]
        activations = activations[0]
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        cam = torch.mul(activations, weights).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return np.zeros((224, 224))
        
    finally:
        handle.remove()

def load_model():
    global model, class_names
    
    checkpoint = torch.load('plant_disease_model_best.pth', map_location=device)
    model = PlantDiseaseModel(num_classes=len(checkpoint['class_names']))
    
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('features.'):
            new_state_dict[k] = v
        elif k.startswith('classifier.'):
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    class_names = checkpoint['class_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.set_grad_enabled(True):
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            num_classes = len(class_names)
            k = min(3, num_classes)
            top_prob, top_indices = torch.topk(probabilities[0], k)
            
            predictions = []
            for prob, idx in zip(top_prob, top_indices):
                predictions.append({
                    'disease': class_names[idx],
                    'probability': float(prob * 100)
                })
            
            grad_cam = generate_gradcam(model, image_tensor)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
            
            original = cv2.imread(filepath)
            original = cv2.resize(original, (224, 224))
            
            overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)
            
            vis_filename = f"vis_{timestamp}.jpg"
            vis_filepath = os.path.join(STATIC_FOLDER, vis_filename)
            cv2.imwrite(vis_filepath, overlay)
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'image_url': f'/uploads/{filename}',
                'grad_cam_url': f'/static/{vis_filename}'
            })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/treatment/<disease>')
def get_treatment(disease):
    language = request.args.get('language', 'en')
    treatments = language_handler.get_treatment_recommendations(disease, language)
    return jsonify(treatments)

@app.route('/soil-analysis')
def soil_analysis():
    return render_template('soil-analysis.html')

@app.route('/plant-disease-detector')
def plant_disease_detector():
    return render_template('plant-disease-detector.html')

app.register_blueprint(soil_bp, url_prefix='/api/soil')

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5001) 