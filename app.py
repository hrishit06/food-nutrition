
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
    print(f"Model type: {type(model).__name__}")
    print(f"Model expects {model.n_features_in_} features")
    print(f"Categorical columns: {cat_cols}")
    print(f"Target columns: {target_cols}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = encoder = cat_cols = target_cols = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    info = {
        'model_loaded': model is not None,
        'n_features': model.n_features_in_ if model else 0,
        'target_cols': target_cols if target_cols else []
    }
    return jsonify(info)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
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
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        print(f"Input data: {input_data}")
        
        # Apply one-hot encoding to categorical columns
        if encoder is not None and cat_cols is not None and len(cat_cols) > 0:
            try:
                categorical_data = input_df[cat_cols]
                print(f"Categorical data: {categorical_data.values}")
                
                # Transform using the fitted encoder - this already returns a numpy array
                encoded_array = encoder.transform(categorical_data)
                print(f"Encoded array shape: {encoded_array.shape}")
                print(f"Encoded array type: {type(encoded_array)}")
                
                # Create column names for encoded features
                encoded_columns = []
                for i, col in enumerate(cat_cols):
                    if hasattr(encoder, 'categories_') and i < len(encoder.categories_):
                        for category in encoder.categories_[i]:
                            encoded_columns.append(f"{col}_{category}")
                
                print(f"Encoded columns: {len(encoded_columns)}")
                
                # Create DataFrame for encoded features
                encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns)
                
                # Drop original categorical columns and add encoded ones
                input_df = input_df.drop(columns=cat_cols)
                input_df = pd.concat([input_df, encoded_df], axis=1)
                
            except Exception as e:
                print(f"Error in encoding: {e}")
                return jsonify({'error': f'Encoding error: {str(e)}'}), 400
        
        # Make prediction
        print(f"Final features shape: {input_df.shape}")
        print(f"Final features columns: {len(input_df.columns)}")
        
        prediction = model.predict(input_df.values)
        print(f"Prediction: {prediction}")
        print(f"Prediction shape: {prediction.shape}")
        
        # Format prediction
        result = {
            'Calories (kcal)': float(round(prediction[0][0], 2)),
            'Protein (g)': float(round(prediction[0][1], 2)),
            'Carbohydrates (g)': float(round(prediction[0][2], 2)),
            'Fats (g)': float(round(prediction[0][3], 2))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


