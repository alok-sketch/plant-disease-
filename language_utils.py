from googletrans import Translator
import json
import os

class LanguageHandler:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'hi': 'Hindi',
            'zh-cn': 'Chinese (Simplified)',
            'ar': 'Arabic',
            'bn': 'Bengali',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'de': 'German'
        }
        
        # Load treatment recommendations
        self.treatment_data = self.load_treatment_data()
    
    def load_treatment_data(self):
        """Load treatment recommendations from JSON file"""
        treatment_file = 'treatment_recommendations.json'
        
        # Default treatment recommendations if file doesn't exist
        default_treatments = {
            "Apple___Apple_scab": {
                "chemical_treatments": [
                    {
                        "name": "Fungicide Application",
                        "products": ["Captan", "Mancozeb", "Sulfur"],
                        "application": "Apply every 7-10 days during growing season",
                        "precautions": "Wear protective gear, avoid application during rain"
                    }
                ],
                "natural_treatments": [
                    {
                        "name": "Neem Oil",
                        "application": "Mix 2-3 tablespoons in 1 gallon of water, spray every 7 days",
                        "benefits": "Natural fungicide, safe for beneficial insects"
                    },
                    {
                        "name": "Baking Soda Solution",
                        "application": "Mix 1 tablespoon baking soda, 1 teaspoon liquid soap in 1 gallon water",
                        "benefits": "Raises leaf pH, inhibits fungal growth"
                    }
                ],
                "prevention": [
                    "Plant resistant varieties",
                    "Improve air circulation through pruning",
                    "Remove fallen leaves and fruit",
                    "Water at the base of trees"
                ]
            }
        }
        
        if os.path.exists(treatment_file):
            with open(treatment_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create the file with default data
            with open(treatment_file, 'w', encoding='utf-8') as f:
                json.dump(default_treatments, f, indent=2)
            return default_treatments
    
    def translate_text(self, text, target_lang='en'):
        """Translate text to target language"""
        try:
            if target_lang == 'en':
                return text
            
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def get_treatment_recommendations(self, disease, language='en'):
        """Get treatment recommendations for a disease in specified language"""
        if disease not in self.treatment_data:
            return {
                'error': self.translate_text('No treatment recommendations available for this disease.', language)
            }
        
        treatment = self.treatment_data[disease]
        
        # Translate all text fields
        translated_treatment = {
            'chemical_treatments': [],
            'natural_treatments': [],
            'prevention': []
        }
        
        # Translate chemical treatments
        for treatment in treatment['chemical_treatments']:
            translated_treatment['chemical_treatments'].append({
                'name': self.translate_text(treatment['name'], language),
                'products': [self.translate_text(product, language) for product in treatment['products']],
                'application': self.translate_text(treatment['application'], language),
                'precautions': self.translate_text(treatment['precautions'], language)
            })
        
        # Translate natural treatments
        for treatment in treatment['natural_treatments']:
            translated_treatment['natural_treatments'].append({
                'name': self.translate_text(treatment['name'], language),
                'application': self.translate_text(treatment['application'], language),
                'benefits': self.translate_text(treatment['benefits'], language)
            })
        
        # Translate prevention measures
        translated_treatment['prevention'] = [
            self.translate_text(measure, language) 
            for measure in treatment['prevention']
        ]
        
        return translated_treatment
    
    def generate_health_report(self, detection_data, language='en'):
        """Generate a comprehensive plant health report"""
        disease = detection_data['disease']
        severity = detection_data['severity']
        confidence = detection_data['confidence']
        
        # Get severity description
        severity_levels = [
            self.translate_text('Mild', language),
            self.translate_text('Moderate', language),
            self.translate_text('Severe', language)
        ]
        
        # Get treatment recommendations
        treatments = self.get_treatment_recommendations(disease, language)
        
        # Generate report
        report = {
            'diagnosis': {
                'disease': self.translate_text(disease, language),
                'severity': severity_levels[severity],
                'confidence': f"{confidence:.1f}%"
            },
            'treatments': treatments,
            'recommendations': [
                self.translate_text('Monitor plant health regularly', language),
                self.translate_text('Follow treatment schedule strictly', language),
                self.translate_text('Document progress with photos', language),
                self.translate_text('Consult local agricultural expert if condition worsens', language)
            ]
        }
        
        return report
    
    def add_treatment_recommendation(self, disease, treatment_data):
        """Add new treatment recommendations to the database"""
        if disease not in self.treatment_data:
            self.treatment_data[disease] = {}
        
        self.treatment_data[disease].update(treatment_data)
        
        # Save to file
        with open('treatment_recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(self.treatment_data, f, indent=2)
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return self.supported_languages 