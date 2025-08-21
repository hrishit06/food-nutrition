# 🍛 Indian Food Nutrition Predictor

A machine learning web application that predicts nutrition values (calories, protein, carbs, fat) for Indian foods using a RandomForestRegressor model.

## Features

- Predicts nutrition values for 85+ Indian foods
- Dropdown menus with all valid categories
- Beautiful responsive web interface
- Real-time predictions

## Model Details

- **Model**: RandomForestRegressor
- **Features**: 135 one-hot encoded features
- **Inputs**: Food Name, Category, Serving Size, Dietary Preference
- **Outputs**: Calories, Protein, Carbohydrates, Fats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/indian-food-nutrition-predictor.git
cd indian-food-nutrition-predictor

### 4. Final `app.py` (cleaned up version):

```python
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load model components
try:
    with open('food_nutrition_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    model = model_dict.get('model')
    encoder = model_dict.get('encoder')
    cat_cols = model_dict.get('cat_cols')
    target_cols = model_dict.get('target_cols')
    
    print("✅ Model loaded successfully!")
    print(f"Model: {type(model).__name__}")
    print(f"Features: {model.n_features_in_}")
    print(f"Targets: {target_cols}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = encoder = cat_cols = target_cols = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    return jsonify({
        'model_loaded': model is not None,
        'n_features': model.n_features_in_ if model else 0,
        'target_cols': target_cols if target_cols else []
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        input_data = data['input_data']
        
        # Validate required fields
        required_fields = ['Food Name', 'Category', 'Serving Size', 'Dietary Preference']
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create input DataFrame and apply one-hot encoding
        input_df = pd.DataFrame([input_data])
        encoded_array = encoder.transform(input_df[cat_cols])
        
        # Create column names for encoded features
        encoded_columns = []
        for i, col in enumerate(cat_cols):
            for category in encoder.categories_[i]:
                encoded_columns.append(f"{col}_{category}")
        
        # Create DataFrame for encoded features
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns)
        input_df = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)
        
        # Make prediction
        prediction = model.predict(input_df.values)
        
        return jsonify({
            'Calories (kcal)': float(round(prediction[0][0], 2)),
            'Protein (g)': float(round(prediction[0][1], 2)),
            'Carbohydrates (g)': float(round(prediction[0][2], 2)),
            'Fats (g)': float(round(prediction[0][3], 2))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
