import data_processing as dp
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = dp.main()
print(df.head())
X = df.drop(columns=['pauseDur'])
y = df['pauseDur']
X_training, X_testing, y_training, y_testing = train_test_split(
    X, y, test_size=0.2
)
# print(f'X_training shape: {X_training.shape}')
# print(f'y_training shape: {y_training.shape}')

# print(X.isna().sum())
# print(y.isna().sum())
# print(X.describe())
print(tf.config.list_physical_devices('GPU'))
# # Create a neural network
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(16, input_shape=(8,), activation="relu"))

model.add(tf.keras.layers.Dense(1))

# Train neural network
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)
model.fit(X_training, y_training, epochs=20,batch_size=8, verbose=2)

# Evaluate how well model performs
model.evaluate(X_testing, y_testing, verbose=2)
