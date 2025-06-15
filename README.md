# 🚗 Vehicle Price Prediction using Machine Learning

This project predicts the price of a vehicle based on various specifications like make, model, year, engine type, fuel type, mileage, transmission, and more. It uses a machine learning pipeline built with TensorFlow and Streamlit for model training and deployment.

---

## 📁 Project Structure

```
vehicle-price-predictor/
│
├── data/
│   └── vehicles.csv               # Dataset
├── model/
│   ├── price_model.h5             # Trained Keras model
│   └── preprocessor.pkl           # Saved preprocessor (scaler + encoder)
├── app.py                         # Streamlit web app
├── main.py                        # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🧐 Model Info

* **Framework:** TensorFlow / Keras
* **Type:** Regression (predicts price in USD)
* **Preprocessing:**

  * Numerical Features: Scaled using `StandardScaler`
  * Categorical Features: Encoded using `OneHotEncoder`
  * Missing values handled using `SimpleImputer`
* **Model Architecture:** Fully connected neural network with 2 hidden layers

---

## 🔍 Features Used

* Year of Manufacture
* Make
* Model
* Cylinders
* Fuel Type
* Mileage
* Transmission
* Trim
* Body Style
* Doors
* Exterior Color
* Interior Color
* Drivetrain

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vehicle-price-predictor.git
cd vehicle-price-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Dataset

Place your dataset as `vehicles.csv` inside the `data/` folder. Make sure it contains all required columns.

### 4. Train the Model

```bash
python main.py
```

This will:

* Preprocess the dataset
* Train a DNN model
* Save the model and preprocessor

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Then open the app in your browser (usually at [http://localhost:8501](http://localhost:8501)).

---

## 📦 Requirements

* Python 3.8+
* TensorFlow
* Pandas
* Scikit-learn
* Streamlit
* Joblib

Install them all with:

```bash
pip install -r requirements.txt
```

---

## 💡 Example Use Cases

* Car dealership pricing tool
* Used vehicle resale platform
* Data science portfolio project
* Price prediction benchmark

---

## 📌 To-Do (Optional Improvements)

* Add image-based price prediction (e.g., using CNN)
* Deploy on Streamlit Cloud or HuggingFace Spaces
* Use more advanced models (XGBoost, LightGBM, etc.)
* Collect real-time data from vehicle APIs

---

## 🧑‍💻 Author

**Yuvraj Singh**
📧 \[[yuvrajsingh8433075079@gmail.com](mailto:yuvrajsingh8433075079@gmail.com)]
🌐 [LinkedIn](https://www.linkedin.com/in/yuvraj-singh-431b7b293/)
💻 Project maintained with ❤️

---

## 📄 License

This project is licensed under the MIT License.
