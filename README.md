# 🌿 AirAware — Smart Air Quality Prediction System

AirAware is a machine-learning web application that predicts the **Air Quality Index (AQI)** from a station ID and pollutant measurements. The app uses a trained regression model and a Streamlit interface to make predictions interactively.

## ✨ Features

- Predict numeric AQI from pollutant measurements.
- Interactive Streamlit user interface.
- Displays the predicted AQI and an AQI category.
- Includes a reference legend for AQI categories used in India.
- Uses a saved model so users do not need to retrain it when opening the app.
- Separate **Home**, **About**, **How It Works**, and **Contact** pages.

## 🧪 Model Inputs

The app expects the following features in the same format and order used during model training:

| Feature | Description |
|---|---|
| `StationId` | Monitoring station identifier |
| `PM2.5` | Fine particulate matter |
| `PM10` | Particulate matter |
| `NO` | Nitric oxide |
| `NO2` | Nitrogen dioxide |
| `NOx` | Nitrogen oxides |
| `NH3` | Ammonia |
| `CO` | Carbon monoxide |
| `SO2` | Sulfur dioxide |
| `O3` | Ozone |
| `Benzene` | Benzene |
| `Toluene` | Toluene |

> **Important:** Use the same units, feature preprocessing, and `StationId` encoding used during training. The application must not pass a string station ID to a model that expects numeric or encoded values. If encoding was used, save and reuse the fitted encoder or include preprocessing in a pipeline.

## 🛠️ Technology Stack

- **Python**
- **Pandas** — input data handling
- **Scikit-learn** — model and evaluation utilities
- **XGBoost** — gradient-boosted trees
- **Streamlit** — interactive web application
- **Pickle** — serializing the trained model

## 📁 Project Structure

```text
AQI_2/
├── app.py
├── model.ipynb
├── aqi_model.pkl
├── requirements.txt
├── city_day_cleaned.csv
├── enriched_aqi_dataset.csv
├── aqi_train_dataset.csv
└── ...
```

Only the files needed to run the app need to be included in a deployment repository. Do not commit virtual environments, secrets, or large datasets unless they are necessary and appropriate to share.

## ⚙️ Setup and Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AQI_2
```

If the project is already on your computer, open its folder in VS Code instead.

### 2. Create and activate a virtual environment (optional but recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

Create a `requirements.txt` file with:

```text
streamlit
pandas
scikit-learn
xgboost
```

Install them:

```bash
python -m pip install -r requirements.txt
```

### 4. Save the trained model

Run the final training cells in `model.ipynb`, then save the trained model and feature names. For the current `app.py`, the pickle file is expected to contain a dictionary with keys `"model"` and `"feature_names"`:

```python
import pickle

model_data = {
    "model": voting_regressor,
    "feature_names": list(x_train.columns)
}

with open("aqi_model.pkl", "wb") as file:
    pickle.dump(model_data, file)
```

Ensure that `aqi_model.pkl` is in the same directory as `app.py`.

### 5. Start Streamlit

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 🚀 Deploy on Streamlit Community Cloud

1. Push `app.py`, `aqi_model.pkl`, and `requirements.txt` to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Choose the repository and set `app.py` as the main file.
5. Deploy the app and test it with valid inputs.

Make sure the model file is within GitHub's file-size limits. If it is too large, use an appropriate model artifact hosting solution and update the loading code.

## 📊 AQI Categories

The interface displays these reference categories:

| AQI range | Category |
|---:|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

These labels are presented as a reference legend. Confirm the target's definition and applicable official AQI standard before using the categories for public health guidance. A numeric model prediction is an estimate, not an official measurement.

## ⚠️ Notes and Limitations

- The default values in the interface are examples, not live sensor measurements.
- Prediction quality depends on representative training data and consistent preprocessing.
- Validate the model on held-out data before using predictions in real-world settings.
- The app does not currently retrieve live pollutant readings.
- Only load pickle files from trusted sources; pickle files can execute code when loaded.
- Keep dependencies compatible with the versions used to train and serialize the model.

## 🔮 Possible Improvements

- Add live AQI and weather data from trusted APIs.
- Store recent predictions and show trends over time.
- Add interactive visualizations and pollutant explanations.
- Package feature preprocessing and the model in a single pipeline.
- Add input validation and tests for unseen or invalid station IDs.

## 👩‍💻 Author

**Project:** AirAware — Smart Air Quality Prediction System

---

*Built as a machine-learning project to explore AQI prediction and interactive deployment with Streamlit.*
