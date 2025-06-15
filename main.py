import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load the dataset
data = pd.read_csv("data/vehicle_data.csv")

# Drop rows with missing target
data = data.dropna(subset=['price'])

# Select features and target
X = data.drop(['price', 'name', 'description'], axis=1)
y = data['price']

# Identify column types
numeric_features = ['year', 'cylinders', 'mileage', 'doors']
categorical_features = ['make', 'model', 'fuel', 'transmission', 'trim', 'body',
                        'exterior_color', 'interior_color', 'drivetrain']

# Preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Save preprocessor for inference
joblib.dump(preprocessor, "model/preprocessor.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: ${mae:.2f}")

# Save the model
model.save("model/price_model.h5")
