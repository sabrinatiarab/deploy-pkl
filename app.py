from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        try:
            prediction = preprocessDataAndPredict(to_predict_list)
            return render_template("predict.html", prediction=prediction)
        except ValueError:
            return "Please enter valid values"

def preprocessDataAndPredict(feature_dict):
    test_data = {k: [v] for k, v in feature_dict.items()}
    test_data = pd.DataFrame(test_data)
    trained_model = joblib.load("lr_model.pkl")
    predict = trained_model.predict(test_data)
    return predict

if __name__ == "__main__":
    app.run(debug=True)