# import complete_data_processing as dp
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# import pandas as pd

# df = pd.read_pickle('expanded_data.pkl')
# X = df.drop(columns=['pauseDur'])
# y = df['pauseDur']
# # X_training, X_testing, y_training, y_testing = train_test_split(
# #     X, y, test_size=0.2
# # )
# # print(f'X_training shape: {X_training.shape}')
# # print(f'y_training shape: {y_training.shape}')

# # print(X.isna().sum())
# # print(y.isna().sum())
# # print(X.describe())
# print(tf.config.list_physical_devices('GPU'))
# # # Create a neural network
# # model = tf.keras.models.Sequential()

# # model.add(tf.keras.layers.Dense(16, input_shape=(8,), activation="relu"))

# # model.add(tf.keras.layers.Dense(1))

# # # Train neural network
# # model.compile(
# #     optimizer="adam",
# #     loss="mean_squared_error",
# #     metrics=["mean_absolute_error"]
# # )
# # model.fit(X_training, y_training, epochs=20,batch_size=8, verbose=2)

# # # Evaluate how well model performs
# # model.evaluate(X_testing, y_testing, verbose=2)
# from sklearn.preprocessing import StandardScaler

# X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_training = scaler.fit_transform(X_training)
# X_testing = scaler.transform(X_testing)

# print(X_training.shape)


# # model = tf.keras.models.Sequential()

# # model.add(tf.keras.layers.Dense(64, input_shape=(X_training.shape[1],), activation="relu"))
# # model.add(tf.keras.layers.Dense(128, activation="relu"))
# # model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer to prevent overfitting
# # model.add(tf.keras.layers.Dense(64, activation="relu"))
# # model.add(tf.keras.layers.Dense(1))

# # # Train neural network
# # model.compile(
# #     optimizer="adam",
# #     loss="mean_squared_error",
# #     metrics=["mean_absolute_error"]
# # )
# # history = model.fit(X_training, y_training, epochs=100, batch_size=8, verbose=2, validation_split=0.2)

# # # Evaluate how well model performs
# # evaluation = model.evaluate(X_testing, y_testing, verbose=2)
# # print(f"Test Loss: {evaluation[0]}, Test MAE: {evaluation[1]}")

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import layers
from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

def attention_block(inputs):
    # Compute attention scores
    attention_probs = Dense(32, activation='softmax')(inputs)
    # Apply attention scores to the inputs
    attention_mul = layers.multiply([inputs, attention_probs])
    return attention_mul

def padding(df):
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.fillna(-9999.0)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(df_numeric.values, padding='post', value=-9999.0, dtype='float32')
    mask = (padded_sequences != -9999.0).astype('float32')
    padded_df = pd.DataFrame(padded_sequences)
    return padded_df, mask

df = df = pd.read_pickle('/Users/madalina/Documents/M1TAL/stage_GC/pro_text/predictions/expanded_data.pkl')

print(df.head())

X = df.drop(columns=['pauseDur'])
y = df['pauseDur']

X_pauses = X.filter(like='pause')
padded_X_pauses, mask_pauses = padding(X_pauses)

X_charBurst = X.filter(like='charBurst')
padded_X_charBurst, mask_charBurst = padding(X_charBurst)

X_pos = X.filter(like='vectors_pos')
padded_X_pos, mask_pos = padding(X_pos)

# X_chunks = X.filter(like='vectors_chunk')
# padded_X_chunks, mask_chunks = padding(X_chunks)

X_pos_start = X.filter(like='pos_start')
padded_X_pos_start, mask_pos_start = padding(X_pos_start)

X_pos_end = X.filter(like='pos_end')
padded_X_pos_end, mask_pos_end = padding(X_pos_end)

X_frequencies = X.filter(like='frequency_').filter(regex='^frequency_[^in_text]')
padded_X_frequencies, mask_frequencies = padding(X_frequencies)

X_frequencies_in_text = X.filter(like='frequency_in_text')
padded_X_frequencies_in_text, mask_frequencies_in_text = padding(X_frequencies_in_text)

X_relative_frequencies = X.filter(like='relative_frequency_')
padded_X_relative_frequencies, mask_relative_grequencies = padding(X_relative_frequencies)

print(padded_X_pauses.shape)
print(padded_X_charBurst.shape)
print(padded_X_pos.shape)
print(padded_X_pos_start.shape)
print(padded_X_pos_end.shape)
print(padded_X_frequencies.shape)
print(padded_X_frequencies_in_text.shape)
print(padded_X_relative_frequencies.shape)

columns_to_drop = (
    list(X.filter(like='pause').columns) +
    list(X.filter(like='charBurst').columns) +
    list(X.filter(like='vectors_pos').columns) +
    list(X.filter(like='vectors_chunk').columns) +
    list(X.filter(like='pos_start').columns) +
    list(X.filter(like='pos_end').columns) +
    list(X.filter(like='frequency_').filter(regex='^frequency_[^in_text]').columns) +
    list(X.filter(like='frequency_in_text').columns) +
    list(X.filter(like='relative_frequency_').columns)
)

X_aggregate = X.drop(columns=columns_to_drop)

# import numpy as np

# # Function to print a few examples of padded sequences and their masks
# import numpy as np
# padded_X_pauses = np.array(padded_X_pauses)
# padded_X_charBurst = np.array(padded_X_charBurst)
# padded_X_pos = np.array(padded_X_pos)
# # padded_X_chunks = np.array(padded_X_chunks)
# padded_X_pos_start = np.array(padded_X_pos_start)
# padded_X_pos_end = np.array(padded_X_pos_end)
# padded_X_frequencies = np.array(padded_X_frequencies)
# padded_X_frequencies_in_text = np.array(padded_X_frequencies_in_text)
# padded_X_relative_frequencies = np.array(padded_X_relative_frequencies)



# import numpy as np

# # Function to print a few examples of padded sequences and their masks
# def print_padded_sequences_and_masks(sequences, mask_value, num_examples=5):
#     for i in range(num_examples):
#         sequence = sequences[i]
#         mask = (sequence != mask_value)
#         print(f"Sequence {i+1}: {sequence}")
#         print(f"Mask {i+1}: {mask}")
#         print()

# # Function to check for non-consecutive False/True values in the mask
# def check_non_consecutive_masks(sequences, mask_value):
#     for i, sequence in enumerate(sequences):
#         mask = (sequence != mask_value)
#         if mask.ndim == 1:
#             mask = np.expand_dims(mask, axis=0)
#         for row in mask:
#             found_false = False
#             for value in row:
#                 if found_false and value:
#                     print(f"Non-consecutive False/True values in sequence {i+1}: {row}")
#                     return False
#                 if not value:
#                     found_false = True
#     return True

# # Example usage
# # Assuming your data is in numpy arrays and padded_X_* are your padded sequences
# padded_sequences = [padded_X_pauses, padded_X_charBurst, padded_X_pos, padded_X_chunks, padded_X_pos_start, padded_X_pos_end, padded_X_frequencies, padded_X_frequencies_in_text, padded_X_relative_frequencies]

# for i, padded_sequence in enumerate(padded_sequences):
#     print(f"Checking sequence {i+1}...")
#     print_padded_sequences_and_masks(padded_sequence, mask_value=-9999.0, num_examples=1)
#     if check_non_consecutive_masks(padded_sequence, mask_value=-9999.0):
#         print(f"Sequence {i+1} is correctly padded and masked.")
#     else:
#         print(f"Sequence {i+1} has padding or masking issues.")

input_pauses = Input(shape=(padded_X_pauses.shape[1], 1))
input_charBurst = Input(shape=(padded_X_charBurst.shape[1], 1))
input_pos = Input(shape=(padded_X_pos.shape[1], 1))
# input_chunks = Input(shape=(padded_X_chunks.shape[1], 1))
input_pos_start = Input(shape=(padded_X_pos_start.shape[1], 1))
input_pos_end = Input(shape=(padded_X_pos_end.shape[1], 1))
input_frequencies = Input(shape=(padded_X_frequencies.shape[1], 1))
input_frequencies_in_text = Input(shape=(padded_X_frequencies_in_text.shape[1], 1))
input_relative_frequencies = Input(shape=(padded_X_relative_frequencies.shape[1], 1))
input_aggregate = Input(shape=(X_aggregate.shape[1],))


# pauses_array = padded_X_pauses.values
# pauses_sequence = pauses_array.reshape(pauses_array.shape[0], pauses_array.shape[1], 1)
# charBurst_array = padded_X_charBurst.values
# charBurst_sequence = charBurst_array.reshape(charBurst_array.shape[0], charBurst_array.shape[1], 1)
# pos_array = padded_X_pos.values
# pos_sequence = pos_array.reshape(pos_array.shape[0], pos_array.shape[1], 1)
# chunks_array = padded_X_chunks.values
# chunks_sequence = chunks_array.reshape(chunks_array.shape[0], chunks_array.shape[1], 1)
# pos_start_array = padded_X_pos_start.values
# pos_start_sequence = pos_start_array.reshape(pos_start_array.shape[0], pos_start_array.shape[1], 1)
# pos_end_array = padded_X_pos_end.values
# pos_end_sequence = pos_end_array.reshape(pos_end_array.shape[0], pos_end_array.shape[1], 1)
# frequencies_array = padded_X_frequencies.values
# frequencies_sequence = frequencies_array.reshape(frequencies_array.shape[0], frequencies_array.shape[1], 1)
# frequencies_in_text_array = padded_X_frequencies_in_text.values
# frequencies_in_text_sequence = frequencies_in_text_array.reshape(frequencies_in_text_array.shape[0], frequencies_in_text_array.shape[1], 1)
# relative_frequencies_array = padded_X_relative_frequencies.values
# relative_frequencies_sequence = relative_frequencies_array.reshape(relative_frequencies_array.shape[0], relative_frequencies_array.shape[1], 1)

masked_pauses = layers.Masking(mask_value=-9999.0)(input_pauses)
masked_charBurst = layers.Masking(mask_value=-9999.0)(input_charBurst)
masked_pos = layers.Masking(mask_value=-9999.0)(input_pos)
# masked_chunks = layers.Masking(mask_value=-9999.0)(input_chunks)
masked_pos_start = layers.Masking(mask_value=-9999.0)(input_pos_start)
masked_pos_end = layers.Masking(mask_value=-9999.0)(input_pos_end)
masked_frequencies = layers.Masking(mask_value=-9999.0)(input_frequencies)
masked_frequencies_in_text = layers.Masking(mask_value=-9999.0)(input_frequencies_in_text)
masked_relative_frequencies = layers.Masking(mask_value=-9999.0)(input_relative_frequencies)


lstm_pauses = LSTM(32, return_sequences=True, use_cudnn=False)(input_pauses)
lstm_charBurst = LSTM(32, return_sequences=True, use_cudnn=False)(input_charBurst)
lstm_pos = LSTM(32, return_sequences=True, use_cudnn=False)(input_pos)
lstm_pos_start = LSTM(32, return_sequences=True, use_cudnn=False)(input_pos_start)
lstm_pos_end = LSTM(32, return_sequences=True, use_cudnn=False)(input_pos_end)
lstm_frequencies = LSTM(32, return_sequences=True, use_cudnn=False)(input_frequencies)
lstm_frequencies_in_text = LSTM(32, return_sequences=True, use_cudnn=False)(input_frequencies_in_text)
lstm_relative_frequencies = LSTM(32, return_sequences=True, use_cudnn=False)(input_relative_frequencies)

# attention_pauses = attention_block(lstm_pauses)
# attention_charBurst = attention_block(lstm_charBurst)
# attention_pos = attention_block(lstm_pos)
# attention_pos_start = attention_block(lstm_pos_start)
# attention_pos_end = attention_block(lstm_pos_end)
# attention_frequencies = attention_block(lstm_frequencies)
# attention_frequencies_in_text = attention_block(lstm_frequencies_in_text)
# attention_relative_frequencies = attention_block(lstm_relative_frequencies)

# # Apply GlobalAveragePooling1D to reduce the sequence dimension
# pooled_pauses = GlobalAveragePooling1D()(lstm_pauses)
# pooled_charBurst = GlobalAveragePooling1D()(lstm_charBurst)
# pooled_pos = GlobalAveragePooling1D()(lstm_pos)
# # pooled_chunks = GlobalAveragePooling1D()(lstm_chunks)
# pooled_pos_start = GlobalAveragePooling1D()(lstm_pos_start)
# pooled_pos_end = GlobalAveragePooling1D()(lstm_pos_end)
# pooled_frequencies = GlobalAveragePooling1D()(lstm_frequencies)
# pooled_frequencies_in_text = GlobalAveragePooling1D()(lstm_frequencies_in_text)
# pooled_relative_frequencies = GlobalAveragePooling1D()(lstm_relative_frequencies)
pooled_pauses = GlobalAveragePooling1D()(lstm_pauses)
pooled_charBurst = GlobalAveragePooling1D()(lstm_charBurst)
pooled_pos = GlobalAveragePooling1D()(lstm_pos)
# pooled_chunks = GlobalAveragePooling1D()(lstm_chunks)
pooled_pos_start = GlobalAveragePooling1D()(lstm_pos_start)
pooled_pos_end = GlobalAveragePooling1D()(lstm_pos_end)
pooled_frequencies = GlobalAveragePooling1D()(lstm_frequencies)
pooled_frequencies_in_text = GlobalAveragePooling1D()(lstm_frequencies_in_text)
pooled_relative_frequencies = GlobalAveragePooling1D()(lstm_relative_frequencies)

# aggregate_sequence = X_aggregate.values.reshape(X_aggregate.shape[0], X_aggregate.shape[1],)
dense_aggregate = Dense(32, activation='relu')(input_aggregate)
concatenated = layers.concatenate([
    pooled_pauses,
    pooled_charBurst,
    pooled_pos,
    pooled_pos_start,
    pooled_pos_end,
    pooled_frequencies,
    pooled_frequencies_in_text,
    pooled_relative_frequencies,
    dense_aggregate
])
dense_out = Dense(128, activation='relu')(concatenated)
output = Dense(1, activation='linear')(dense_out)

model = Model(inputs=[input_pauses, input_charBurst, input_pos, input_pos_start, input_pos_end, input_frequencies, input_frequencies_in_text, input_relative_frequencies, input_aggregate], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Split your data for training and testing
X_train_pauses, X_test_pauses = train_test_split(padded_X_pauses, test_size=0.2, random_state=random_seed)
X_train_charBurst, X_test_charBurst = train_test_split(padded_X_charBurst, test_size=0.2, random_state=random_seed)
X_train_pos, X_test_pos = train_test_split(padded_X_pos, test_size=0.2, random_state=random_seed)
X_train_pos_start, X_test_pos_start = train_test_split(padded_X_pos_start, test_size=0.2, random_state=random_seed)
X_train_pos_end, X_test_pos_end = train_test_split(padded_X_pos_end, test_size=0.2, random_state=random_seed)
X_train_frequencies, X_test_frequencies = train_test_split(padded_X_frequencies, test_size=0.2, random_state=random_seed)
X_train_frequencies_in_text, X_test_frequencies_in_text = train_test_split(padded_X_frequencies_in_text, test_size=0.2, random_state=random_seed)
X_train_relative_frequencies, X_test_relative_frequencies = train_test_split(padded_X_relative_frequencies, test_size=0.2, random_state=random_seed)
X_train_aggregate, X_test_aggregate = train_test_split(X_aggregate, test_size=0.2, random_state=random_seed)

# Split target into training and testing
y_train, y_test = train_test_split(y, test_size=0.2, random_state=random_seed)

model.fit([X_train_pauses, X_train_charBurst, X_train_pos, X_train_pos_start, X_train_pos_end, X_train_frequencies, X_train_frequencies_in_text, X_train_relative_frequencies, X_train_aggregate], y_train, epochs=10, batch_size=32, verbose=2, validation_split=0.2)
model.evaluate([X_test_pauses, X_test_charBurst, X_test_pos, X_test_pos_start, X_test_pos_end, X_test_frequencies, X_test_frequencies_in_text, X_test_relative_frequencies, X_test_aggregate], y_test, verbose=2)

y_pred = model.predict([X_test_pauses, X_test_charBurst, X_test_pos, X_test_pos_start, X_test_pos_end, X_test_frequencies, X_test_frequencies_in_text, X_test_relative_frequencies, X_test_aggregate])

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2}")
#embedding = layers.Embedding(input_dim=116517, output_dim=128, mask_zero=True)
#masked_output = embedding(padded_X)
# masking_layer = layers.Masking()

#model = Sequential()
#model.add(Masking(mask_value=-1, input_shape=(padded_X.shape[1],))).