from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('penguin_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        bill_length = float(request.form['bill_length_mm'])
        bill_depth = float(request.form['bill_depth_mm'])
        flipper_length = float(request.form['flipper_length_mm'])
        body_mass = float(request.form['body_mass_g'])

        # One-hot encode 'island'
        island = request.form['island']
        island_dream = 0
        island_torgersen = 0
        if island == 'Dream':
            island_dream = 1
        elif island == 'Torgersen':
            island_torgersen = 1

        # One-hot encode 'sex' (sex_male: 1 if male, 0 if female)
        sex = request.form['sex']
        sex_male = 1 if sex == 'male' else 0

        
        # Combine all features into a single NumPy array
        features = np.array([[bill_length, bill_depth, flipper_length, body_mass, island_dream, island_torgersen, sex_male]])
        
        print(features)

        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Mapping prediction to species names
        species_mapping = {1: 'Adelie', 2: 'Gentoo', 3: 'Chinstrap'}
        predicted_species = species_mapping.get(prediction, "Unknown")
        
        return render_template('index.html', prediction_text=f'Predicted Penguin Species: {predicted_species}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
