import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import json
from datetime import datetime
import sqlite3
from geopy.geocoders import Nominatim
import folium
from folium import plugins

class CameraHandler:
    def __init__(self, model, transform, class_names, offline_mode=False):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.offline_mode = offline_mode
        self.camera = None
        self.geolocator = Nominatim(user_agent="plant_disease_detector")
        
        # Initialize database for offline storage
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for offline storage"""
        conn = sqlite3.connect('plant_disease_data.db')
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     disease TEXT,
                     severity INTEGER,
                     confidence REAL,
                     location TEXT,
                     image_path TEXT,
                     feedback TEXT)''')
        
        conn.commit()
        conn.close()
    
    def start_camera(self, camera_id=0):
        """Start the camera for real-time detection"""
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
    
    def stop_camera(self):
        """Stop the camera"""
        if self.camera:
            self.camera.release()
    
    def get_frame(self):
        """Get a frame from the camera and process it"""
        if not self.camera:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform image
        image_tensor = self.transform(pil_image).unsqueeze(0)
        
        return frame_rgb, image_tensor
    
    def process_frame(self, image_tensor):
        """Process a frame and return predictions"""
        with torch.no_grad():
            # Get disease and severity predictions
            disease_output, severity_output = self.model(image_tensor)
            
            # Get disease probabilities
            disease_probs = torch.nn.functional.softmax(disease_output, dim=1)
            _, predicted = disease_probs.max(1)
            
            # Get severity probabilities
            severity_probs = torch.nn.functional.softmax(severity_output, dim=1)
            severity_level, severity_conf = self.model.estimate_severity(image_tensor)
            
            # Get Grad-CAM visualization
            grad_cam = self.model.get_grad_cam(image_tensor, predicted.item())
            
            return {
                'disease': self.class_names[predicted.item()],
                'confidence': float(disease_probs[0, predicted].item() * 100),
                'severity': severity_level,
                'severity_confidence': float(severity_conf * 100),
                'grad_cam': grad_cam
            }
    
    def save_detection(self, detection, image_path, location=None):
        """Save detection to database"""
        conn = sqlite3.connect('plant_disease_data.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO detections
                    (timestamp, disease, severity, confidence, location, image_path)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (datetime.now().isoformat(),
                  detection['disease'],
                  detection['severity'],
                  detection['confidence'],
                  json.dumps(location) if location else None,
                  image_path))
        
        conn.commit()
        conn.close()
    
    def get_location(self):
        """Get current location using IP-based geolocation"""
        try:
            location = self.geolocator.geocode("me")
            if location:
                return {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'address': location.address
                }
        except Exception as e:
            print(f"Error getting location: {e}")
        return None
    
    def create_disease_map(self):
        """Create a map showing disease outbreaks"""
        conn = sqlite3.connect('plant_disease_data.db')
        c = conn.cursor()
        
        # Get all detections with location
        c.execute('''SELECT disease, location, timestamp FROM detections 
                    WHERE location IS NOT NULL''')
        detections = c.fetchall()
        
        # Create map centered on the first detection or default location
        if detections:
            first_loc = json.loads(detections[0][1])
            center_lat, center_lon = first_loc['latitude'], first_loc['longitude']
        else:
            center_lat, center_lon = 0, 0
        
        disease_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Add markers for each detection
        for disease, location, timestamp in detections:
            loc = json.loads(location)
            folium.Marker(
                [loc['latitude'], loc['longitude']],
                popup=f"{disease}<br>Detected: {timestamp}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(disease_map)
        
        # Add heatmap
        heat_data = [[json.loads(loc[1])['latitude'], 
                     json.loads(loc[1])['longitude']] 
                    for loc in detections]
        plugins.HeatMap(heat_data).add_to(disease_map)
        
        # Save map
        map_path = 'static/disease_map.html'
        disease_map.save(map_path)
        
        conn.close()
        return map_path
    
    def collect_feedback(self, detection_id, feedback):
        """Collect user feedback for a detection"""
        conn = sqlite3.connect('plant_disease_data.db')
        c = conn.cursor()
        
        c.execute('''UPDATE detections SET feedback = ? WHERE id = ?''',
                 (feedback, detection_id))
        
        conn.commit()
        conn.close()
    
    def export_data(self, format='json'):
        """Export detection data for model improvement"""
        conn = sqlite3.connect('plant_disease_data.db')
        c = conn.cursor()
        
        c.execute('''SELECT * FROM detections''')
        detections = c.fetchall()
        
        if format == 'json':
            data = []
            for det in detections:
                data.append({
                    'id': det[0],
                    'timestamp': det[1],
                    'disease': det[2],
                    'severity': det[3],
                    'confidence': det[4],
                    'location': json.loads(det[5]) if det[5] else None,
                    'image_path': det[6],
                    'feedback': det[7]
                })
            
            with open('exported_data.json', 'w') as f:
                json.dump(data, f, indent=2)
        
        conn.close()
        return 'exported_data.json' 