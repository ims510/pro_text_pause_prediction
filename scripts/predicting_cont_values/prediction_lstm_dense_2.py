
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import layers
from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split

def padding(df):
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.fillna(-9999.0)
    padded_df = tf.keras.preprocessing.sequence.pad_sequences(df_numeric.values, padding='post', value=-9999.0, dtype='float32')
    padded_df = pd.DataFrame(padded_df)
    return padded_df

df = df = pd.read_pickle('expanded_data.pkl')
# print(df.head())

X = df.drop(columns=['pauseDur'])
y = df['pauseDur']

X_pauses = X.filter(like='pause')
padded_X_pauses = padding(X_pauses)

X_charBurst = X.filter(like='charBurst')
padded_X_charBurst = padding(X_charBurst)

X_pos = X.filter(like='vectors_pos')
padded_X_pos = padding(X_pos)

# X_chunks = X.filter(like='vectors_chunk')
# padded_X_chunks = padding(X_chunks)

X_pos_start = X.filter(like='pos_start')
padded_X_pos_start = padding(X_pos_start)

X_pos_end = X.filter(like='pos_end')
padded_X_pos_end = padding(X_pos_end)

X_frequencies = X.filter(like='frequency_').filter(regex='^frequency_[^in_text]')
padded_X_frequencies = padding(X_frequencies)

X_frequencies_in_text = X.filter(like='frequency_in_text')
padded_X_frequencies_in_text = padding(X_frequencies_in_text)

X_relative_frequencies = X.filter(like='relative_frequency_')
padded_X_relative_frequencies = padding(X_relative_frequencies)

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


lstm_pauses = LSTM(16, return_sequences=True, use_cudnn=False)(masked_pauses)
lstm_charBurst = LSTM(16, return_sequences=True, use_cudnn=False)(masked_charBurst)
lstm_pos = LSTM(16, return_sequences=True, use_cudnn=False)(masked_pos)
# lstm_chunks = LSTM(16, return_sequences=True, use_cudnn=False)(masked_chunks)
lstm_pos_start = LSTM(16, return_sequences=True, use_cudnn=False)(masked_pos_start)
lstm_pos_end = LSTM(16, return_sequences=True, use_cudnn=False)(masked_pos_end)
lstm_frequencies = LSTM(16, return_sequences=True, use_cudnn=False)(masked_frequencies)
lstm_frequencies_in_text = LSTM(16, return_sequences=True, use_cudnn=False)(masked_frequencies_in_text)
lstm_relative_frequencies = LSTM(16, return_sequences=True, use_cudnn=False)(masked_relative_frequencies)

# Apply GlobalAveragePooling1D to reduce the sequence dimension
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
dense_aggregate = Dense(16, activation='relu')(input_aggregate)

# concatenated = layers.concatenate([lstm_pauses, lstm_charBurst, lstm_pos, lstm_pos_start, lstm_pos_end, lstm_frequencies, lstm_frequencies_in_text, dense_aggregate])
concatenated = layers.concatenate([pooled_pauses, pooled_charBurst, pooled_pos, pooled_pos_start, pooled_pos_end, pooled_frequencies, pooled_frequencies_in_text, pooled_relative_frequencies, dense_aggregate])
dense_out = Dense(16, activation='relu')(concatenated)
output = Dense(1, activation='linear')(dense_out)

model = Model(inputs=[input_pauses, input_charBurst, input_pos, input_pos_start, input_pos_end, input_frequencies, input_frequencies_in_text, input_relative_frequencies, input_aggregate], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# split padded data into test and train
X_train_pauses, X_test_pauses = train_test_split(padded_X_pauses, test_size=0.2, random_state=42)
X_train_charBurst, X_test_charBurst = train_test_split(padded_X_charBurst, test_size=0.2, random_state=42)
X_train_pos, X_test_pos = train_test_split(padded_X_pos, test_size=0.2, random_state=42)
# X_train_chunks, X_test_chunks = train_test_split(padded_X_chunks, test_size=0.2, random_state=42)
X_train_pos_start, X_test_pos_start = train_test_split(padded_X_pos_start, test_size=0.2, random_state=42)
X_train_pos_end, X_test_pos_end = train_test_split(padded_X_pos_end, test_size=0.2, random_state=42)
X_train_frequencies, X_test_frequencies = train_test_split(padded_X_frequencies, test_size=0.2, random_state=42)
X_train_frequencies_in_text, X_test_frequencies_in_text = train_test_split(padded_X_frequencies_in_text, test_size=0.2, random_state=42)
X_train_relative_frequencies, X_test_relative_frequencies = train_test_split(padded_X_relative_frequencies, test_size=0.2, random_state=42)
X_train_aggregate, X_test_aggregate = train_test_split(X_aggregate, test_size=0.2, random_state=42)

# split target into test and train
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

model.fit([X_train_pauses, X_train_charBurst, X_train_pos, X_train_pos_start, X_train_pos_end, X_train_frequencies, X_train_frequencies_in_text, X_train_relative_frequencies, X_train_aggregate], y_train, epochs=10, batch_size=32, verbose=2, validation_split=0.2)
model.evaluate([X_test_pauses, X_test_charBurst, X_test_pos, X_test_pos_start, X_test_pos_end, X_test_frequencies, X_test_frequencies_in_text, X_test_relative_frequencies, X_test_aggregate], y_test, verbose=2)
#embedding = layers.Embedding(input_dim=116517, output_dim=128, mask_zero=True)
#masked_output = embedding(padded_X)
# masking_layer = layers.Masking()

#model = Sequential()
#model.add(Masking(mask_value=-1, input_shape=(padded_X.shape[1],)))