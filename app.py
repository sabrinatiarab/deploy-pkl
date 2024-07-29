from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
multi_output_regressor = joblib.load('rmo_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Convert form data to DataFrame and handle missing or incorrect values
        input_data = {}
        for key, value in form_data.items():
            try:
                input_data[key] = [float(value)]
            except ValueError:
                return jsonify({"error": f"Invalid value for {key}: {value}. Please enter a valid number."}), 400

        input_df = pd.DataFrame.from_dict(input_data)

        # Make predictions
        predictions = multi_output_regressor.predict(input_df)
        predicted_layanan = predictions[0, 0]
        predicted_biaya_sewa = predictions[0, 1]

        return render_template('predict.html',
                               predicted_layanan=predicted_layanan,
                               predicted_biaya_sewa=predicted_biaya_sewa)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)