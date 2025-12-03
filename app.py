from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model + encoder
try:
    model = joblib.load("aqi_model_high_accuracy.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Model loaded.")
except:
    model = None
    label_encoder = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        def get_float(val):
            try:
                if val is None or val == "":
                    return np.nan
                return float(val)
            except:
                return np.nan

        data = {
            "City": request.form.get("city"),   # MATCHES HTML

            "PM2.5": get_float(request.form.get("pm25")),
            "PM10": get_float(request.form.get("pm10")),
            "NO": get_float(request.form.get("no")),
            "NO2": get_float(request.form.get("no2")),
            "NOx": get_float(request.form.get("nox")),
            "NH3": get_float(request.form.get("nh3")),
            "CO": get_float(request.form.get("co")),
            "SO2": get_float(request.form.get("so2")),
            "O3": get_float(request.form.get("o3")),
            "Benzene": get_float(request.form.get("benzene")),
            "Toluene": get_float(request.form.get("toluene")),
            "Xylene": get_float(request.form.get("xylene")),
           
           
        }

        df_input = pd.DataFrame([data])

        pred_index = model.predict(df_input)[0]
        category = label_encoder.inverse_transform([pred_index])[0]

        return render_template("index.html", result=category)

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
