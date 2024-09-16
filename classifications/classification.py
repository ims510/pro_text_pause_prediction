import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
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
    df = pd.read_pickle('/Users/madalina/Documents/M1TAL/stage_GC/pro_text/predictions/expanded_data.pkl')

    df = clean_df(df)
    df = padding(df)

    print(df.head())

    X = df.drop('pauseDur', axis=1)  # Features
    y = df['pauseDur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    # print(y_train)
    y_train = disc.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1)
    y_test = disc.fit_transform(y_test.values.reshape(-1, 1)).reshape(-1)
    # print(y_train)
    print("Bin edges for 'pauseDur':")
    for i, edge in enumerate(disc.bin_edges_[0]):
        if i < len(disc.bin_edges_[0]) - 1:
            print(f"Bin {i}: {edge} to {disc.bin_edges_[0][i + 1]}")
    
     # Build the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy}")

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

if __name__ == '__main__':
    main()