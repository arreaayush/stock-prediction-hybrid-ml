# 📈 Stock Market Prediction with ML (LSTM + RF + Sentiment)

A full-fledged stock prediction web application using a **hybrid ML model** combining **LSTM** and **Random Forest**, enriched with **sentiment analysis** from financial news sources.

## 🔍 Overview

This project aims to predict stock prices and trends based on historical stock data and real-time news sentiment. It leverages the power of:

- **LSTM (Long Short-Term Memory)**: For learning patterns in sequential stock data.
- **Random Forest Regressor**: For handling complex relationships in data with ensemble learning.
- **Hybrid Model**: A weighted average of both LSTM and RF outputs for more accurate prediction.
- **Sentiment Analysis**: Extracting sentiment polarity from financial headlines using VADER.

The model is trained in a Jupyter Notebook and deployed using a **Streamlit Web App**.

---

## 🛠 Prerequisites & Setup

To run this project locally, ensure you have the following installed:

### ✅ Requirements

- Python 3.9+
- pip

### 📦 Install Dependencies

Install the required Python packages using the following:

```bash
pip install -r requirements.txt
```

> A `requirements.txt` file is included with packages like `pandas`, `numpy`, `alpha_vantage`, `scikit-learn`, `tensorflow`, `nltk`, `streamlit`, `matplotlib`, `plotly`, `requests`, etc.

---

## 🧪 Model Training (Jupyter Notebook)

If you want to train the models from scratch:

1. Run `stock_model_train.ipynb` to:
   - Fetch stock data (e.g., `MSFT`)
   - Preprocess and scale the data
   - Train the LSTM and Random Forest models
   - Save the models and scaler using `joblib` and `h5py`

2. Outputs:
   - `lstm_model.keras`
   - `rf_model.pkl`
   - `scaler.pkl`

---

## 🖥 Streamlit App Usage

To launch the web application:

```bash
streamlit run app.py
```

### App Sections

- **Select Stock**: Default set to `MSFT`, can be made dynamic in future.
- **Last & First 5 Rows**: Display the last & first few rows of available data.
- **Model Predictions**: Show predictions by LSTM, Random Forest, and Hybrid.
- **Charts**: Visualize technical indicators and prediction performance.
- **Sentiment Analysis**: Show recent financial news with sentiment score.

---

## 📁 Folder Structure

```
├── app.py                  # Streamlit Web App
├── stock_model_train.ipynb # Jupyter Notebook for model training
├── lstm_model.keras        # Saved LSTM model
├── rf_model.pkl            # Saved Random Forest model
├── scaler.pkl              # Saved MinMaxScaler
├── requirements.txt        # Python dependencies
├── utils/                  # (Optional) Helper functions
└── README.md               # Project Documentation
```

---

## 📌 Future Improvements

- [*] Add dynamic stock ticker input via dropdown
- [*] Improve hybrid model weighting with optimization
- [*] Add more sentiment sources (e.g., Reddit, Twitter)
- [*] Deploy on cloud (e.g., Streamlit Cloud, Heroku)

---

## 👨‍💻 Author

**Aayush Shinde**  
_Data Analyst | ERP Technical Consultant 
📧 Reach out on [LinkedIn](https://www.linkedin.com/in/aayush-shinde-a809a0251/) or [GitHub](https://github.com/arreaayush)

---

## 📝 License

This project is licensed under the MIT License - feel free to use and modify for educational purposes.
