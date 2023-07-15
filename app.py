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
        except Exception as e:
            return f"An error occurred: {str(e)}"

def preprocessDataAndPredict(feature_dict):
    test_data = {k: [v] for k, v in feature_dict.items()}
    test_data = pd.DataFrame(test_data)
    trained_model = joblib.load("lr_model.pkl")
    predict = trained_model.predict(test_data)
    
    if predict == 0:
        prediction = "Class 0"
    elif predict == 1:
        prediction = "Class 1"
    elif predict == 2:
        prediction = "Class 2"
    elif predict == 3:
        prediction = "Class 3"
    elif predict == 4:
        prediction = "Class 4"
    else:
        prediction = "Class 5"
    
    return prediction

if __name__ == "__main__":
    app.run(debug=True)