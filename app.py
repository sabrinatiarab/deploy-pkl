from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("mo_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            features = ['Bidang_Baku', 'Tipe', 'Bandwidth', 'Kabupaten/Kota', 'Wilayah']
            data = [int(request.form[feature]) for feature in features]  # Convert to float
            prediction = model.predict([data])
            predicted_layanan = prediction[0][0]  # Access the predicted Layanan
            predicted_biaya_sewa = prediction[0][1]  # Access the predicted Biaya_Sewa
            return render_template('predict.html', predicted_biaya_sewa=predicted_biaya_sewa, predicted_layanan=predicted_layanan)
        except Exception as e:
            return f"An error occurred: {str(e)}"

def preprocessDataAndPredict(feature_dict):
    # Make sure the keys in feature_dict match the feature names used during training
    required_features = ["Bidang_Baku", "Tipe", "Bandwidth", "Kabupaten/Kota", "Wilayah"]  # Add other required features

    for feature in required_features:
        if feature not in feature_dict:
            raise ValueError(f"Missing feature: {feature}")

    # Preprocess the data (convert to DataFrame, handle categorical variables, etc.)
    data = pd.DataFrame([feature_dict])

    # Make the prediction using the loaded model
    predicted_values = model.predict(data)

    # Replace this with your actual post-processing logic to extract predictions
    predicted_layanan = predicted_values[0][0]  # Access the predicted Layanan
    predicted_biaya_sewa = predicted_values[0][1]  # Access the predicted Biaya_Sewa

    prediction = {
        "Layanan": predicted_layanan,
        "Biaya_Sewa": predicted_biaya_sewa
    }

    return prediction

if __name__ == "__main__":
    app.run(debug=True)