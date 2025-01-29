import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
    
    df = pd.read_pickle('../../data/input/expanded_data_decale.pkl')

    df = clean_df(df)
    df = padding(df)

    print(df.head())

    X = df.drop('pauseDur', axis=1)  # Features
    y = df['pauseDur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    
    # Fit the discretizer on y_train and transform y_train and y_test
    y_train = disc.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1)
    y_test = disc.fit_transform(y_test.values.reshape(-1, 1)).reshape(-1)
    
    # Print the bin edges
    print("Bin edges for 'pauseDur':")
    for i, edge in enumerate(disc.bin_edges_[0]):
        if i < len(disc.bin_edges_[0]) - 1:
            print(f"Bin {i}: {edge} to {disc.bin_edges_[0][i + 1]}")
    
    # Build the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))  # Add dropout with 50% rate
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))  # Add dropout with 50% rate
    model.add(Dense(5, activation='softmax'))  # 5 output classes

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with early stopping
    model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.2, callbacks=[early_stopping])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(5))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    main()
