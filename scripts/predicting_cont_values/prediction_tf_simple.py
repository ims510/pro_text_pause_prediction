
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Set to '0' for debug, '1' for info, '2' for warning, '3' for error
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def clean_df(df):
    columns_to_drop = (
        list(df.filter(like='vectors_pos').columns) +
        list(df.filter(like='vectors_chunk').columns) +
        list(df.filter(like='pos_start').columns) +
        list(df.filter(like='pos_end').columns) +
        list(df.filter(like='frequency_').filter(regex='^frequency_[^in_text]').columns) +
        list(df.filter(like='frequency_in_text').columns) +
        list(df.filter(like='relative_frequency_').columns) +
        list(df.filter(like='totalActions').columns) +
        list(df.filter(like='totalChars').columns) +
        list(df.filter(like='finalChars').columns) +
        list(df.filter(like='totalDeletions').columns) +
        list(df.filter(like='innerDeletions').columns) +
        list(df.filter(like='docLen').columns) +
        list(df.filter(like='num_actions').columns)
    )

    basic_df = df.drop(columns=columns_to_drop)
    print(basic_df.columns)
    return basic_df


def padding(df):
    # Save the original column names
    original_columns = df.columns

    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.fillna(0)
    padded_df = tf.keras.preprocessing.sequence.pad_sequences(df_numeric.values, padding='post', value=-9999.0, dtype='float32')

    # Convert back to DataFrame and reassign original column names
    padded_df = pd.DataFrame(padded_df, columns=original_columns)
    padded_df.to_csv('padded_data.csv')
    return padded_df

def main():
    df = pd.read_pickle('expanded_data.pkl')

    # df = clean_df(df)
    df = padding(df)

    print(df.head())

    X = df.drop('pauseDur', axis=1)  # Features
    y = df['pauseDur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential()

    # Input layer
    model.add(Input(shape=(X_train.shape[1],)))

    # Hidden layers
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='relu'))  # Output layer for regression


    # Compile the model with a lower learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Print the model summary
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with a smaller batch size
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2, callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Mean Absolute Error on Test Set: {mae:.4f}")
    print(f"Loss on Test Set: {loss:.4f}")

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R² Score on Test Set: {r2:.4f}")
    for i in range(10):
        print(f"Actual: {y_test.values[i]}, Predicted: {y_pred[i][0]}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Values')
    plt.plot(y_pred, label='Predicted Values', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Pause Duration')
    plt.title('Actual vs Predicted Pause Duration')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()