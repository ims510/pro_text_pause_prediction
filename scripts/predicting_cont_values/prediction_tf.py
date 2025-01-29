import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Adjust TensorFlow log level
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from scipy.stats import zscore

print("Loading data...")
# Load the data
pickle_file = 'expanded_data_decale.pkl'
df = pd.read_pickle(pickle_file)
correlations = df.corr()
# print(correlations['pauseDur'].sort_values(ascending=False))
# Features and target variable
X = df.drop('pauseDur', axis=1)  # Features
y = df['pauseDur']

# Log-transform and clip target variable
y_transformed = np.log(y + 1)  # Log transform to normalize
y_clipped = np.clip(y_transformed, y_transformed.quantile(0.01), y_transformed.quantile(0.99))  # Remove extreme outliers
y = y_clipped

print(f"Mean of y: {y.mean()}")
print(f"Standard deviation of y: {y.std()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert non-numeric columns to numeric
non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
print(f"Non-numeric columns: {non_numeric_columns}")
for col in non_numeric_columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill NaN values with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Remove outliers in the target variable based on z-scores
z_scores = np.abs(zscore(y_train))  # Calculate z-scores
mask = z_scores < 3  # Identify rows without outliers
mask = pd.Series(mask, index=X_train.index)  # Align mask with X_train index

# Apply the mask to clean X_train and y_train
X_train_cleaned = X_train[mask]
y_train_cleaned = y_train[mask]

# Scale target variable
scaler_y = StandardScaler()
y_train_cleaned_scaled = scaler_y.fit_transform(y_train_cleaned.values.reshape(-1, 1))

# Redo scaling for cleaned X_train
X_train_cleaned_scaled = scaler_X.transform(X_train_cleaned)

# Define the neural network model
model = Sequential()
model.add(Input(shape=(X_train_cleaned_scaled.shape[1],)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("Starting training...")
# Train the model
history = model.fit(X_train_cleaned_scaled, y_train_cleaned_scaled, validation_split=0.2, epochs=200, batch_size=32, verbose=2)

# Make predictions on the test set
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
# Flatten the arrays to ensure they are 1D
y_pred = y_pred.ravel()
y_test = y_test.ravel()

# Evaluate the model
mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error on the original scale
r2 = r2_score(y_test, y_pred)  # R^2 score on the original scale

print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
# Evaluate the model
mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error on the original scale
r2 = r2_score(y_test, y_pred)  # R^2 score on the original scale

print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot true vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("True Values (Original Scale)")
plt.ylabel("Predictions (Original Scale)")
plt.title("True vs. Predicted Values")
plt.legend()
plt.show()

# Plot training & validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
