import csv
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from scipy.stats import iqr

csv_file = "/Users/madalina/Documents/M1TAL/stage_GC/pro_text/predictions/all_tagged.csv"
# csv_file = "/Users/madalina/Documents/M1TAL/stage_GC/pro_text/predictions/vectorised_data.csv"

def process_row(row, data):
    if row[3] not in data:
        data[row[3]] = [row[i] for i in range(4, 15)]
        data[row[3]].append(row[17])
        data[row[3]].append(row[20])
        # data[row[3]].append(row[18]) # this is the type of burst (P, R, ER) 
        char_burst = [(row[15], row[16], row[19], row[22], row[23], row[24], row[25], row[26])]
        data[row[3]].append(char_burst)
    else:
        char_burst = (row[15], row[16], row[19], row[22], row[23], row[24], row[25], row[26])
        data[row[3]][-1].append(char_burst)
    return data

def process_csv(reader):
    people = {}
    for row in reader:
        if row[0] not in people:
            data = {}
            data = process_row(row, data)
            people[row[0]] = data
        else:
            data = people[row[0]]
            data = process_row(row, data)
            people[row[0]] = data
    return people

def read_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        people = process_csv(reader)
    return people

def extract_features(burst_data):
    burstStart = float(burst_data[0])
    burstDur = float(burst_data[1])
    pauseDur = float(burst_data[2])  # This is the target we want to predict
    cycleDur = float(burst_data[3])
    burstPct = float(burst_data[4])
    pausePct = float(burst_data[5])
    totalActions = int(burst_data[6])
    totalChars = int(burst_data[7])
    finalChars = int(burst_data[8])
    totalDeletions = int(burst_data[9])
    innerDeletions = int(burst_data[10])
    docLen = int(burst_data[11])
    
    pauses = burst_data[12]
    charBursts = burst_data[13]  # This is the list of tuples
    # print(charBursts)
    # print("##################")

    # Feature extraction from charBurst
    # avg_shift = sum(abs(int(end) - int(start)) for start, end, _ in charBursts) / len(charBursts) # not sure if good metric
    num_actions = len(charBursts)
    
    return {
        # 'burstStart': burstStart,
        # 'burstDur': burstDur,
        # 'cycleDur': cycleDur,
        # 'burstPct': burstPct,
        # 'pausePct': pausePct,
        'totalActions': totalActions,
        'totalChars': totalChars,
        'finalChars': finalChars,
        'totalDeletions': totalDeletions,
        'innerDeletions': innerDeletions,
        'docLen': docLen,
        # 'avg_shift': avg_shift,
        'num_actions': num_actions,
        'charBursts': charBursts,
        'pauses': pauses,
        'pauseDur': pauseDur # This is the target we want to predict
    }


def vectorise_text(df):
    charBursts = []
    poss = []
    chunkss = []
    for i in range(len(df)):
        for j in range(len(df['charBursts'][i])):

            charBurst = df['charBursts'][i][j][2]
            pos = df['charBursts'][i][j][3]
            chunks = df['charBursts'][i][j][4]
            charBursts.append(charBurst.split())
            poss.append(pos.split(","))
            chunkss.append(chunks.split(","))
            
    model_charBursts = Word2Vec(sentences=charBursts, vector_size=50, window=5, min_count=1, workers=4)
    model_pos = Word2Vec(sentences=poss, vector_size=50, window=5, min_count=1, workers=4)
    model_chunks = Word2Vec(sentences=chunkss, vector_size=50, window=5, min_count=1, workers=4)

    big_list_vectors_charBursts = []
    big_list_vectors_pos = []
    big_list_vectors_chunks = []
    big_list_pos_start = []
    big_list_pos_end = []
    big_list_frequencies = []
    big_list_frequencies_in_text = []
    big_list_relative_frequencies = []
    for i in range(len(df)):
        list_vectors_charBursts = []
        list_pauses = []
        list_vectors_pos = []
        list_vectors_chunks = []
        list_pos_start = []
        list_pos_end = []
        list_frequencies = []
        list_frequencies_in_text = []
        list_relative_frequencies = []
        for j in range(len(df['charBursts'][i])):
            # print(df['charBursts'][i][j])   
            charBurst = df['charBursts'][i][j][2]
            pos = df['charBursts'][i][j][3]
            chunks = df['charBursts'][i][j][4]

            vectors_charBurst = [model_charBursts.wv[word] for word in charBurst.split()]
            vectors_pos = [model_pos.wv[word] for word in pos.split(",")]
            vectors_chunks = [model_chunks.wv[word] for word in chunks.split(",")]
            
            list_vectors_charBursts.extend(vectors_charBurst)
            list_vectors_pos.extend(vectors_pos)
            list_vectors_chunks.extend(vectors_chunks)
            list_pos_start.append(int(df['charBursts'][i][j][0]))
            list_pos_end.append(int(df['charBursts'][i][j][1]))
            list_frequencies.extend([int(f_value) if f_value != '' else 0 for f_value in df['charBursts'][i][j][5].split(",")])
            list_frequencies_in_text.extend([int(fit_value) if fit_value != '' else 0 for fit_value in df['charBursts'][i][j][6].split(",")])
            list_relative_frequencies.extend([float(rf_value) if rf_value != '' else 0 for rf_value in df['charBursts'][i][j][7].split(",")])
        
        big_list_vectors_charBursts.append(list_vectors_charBursts)
        big_list_vectors_pos.append(list_vectors_pos)
        big_list_vectors_chunks.append(list_vectors_chunks)
        big_list_pos_start.append(list_pos_start)
        big_list_pos_end.append(list_pos_end)
        big_list_frequencies.append(list_frequencies)
        big_list_frequencies_in_text.append(list_frequencies_in_text)
        big_list_relative_frequencies.append(list_relative_frequencies)
            # df["vectors_charBurst"]
            # df.at[i, 'charBursts'][j] = (df['charBursts'][i][j][0], df['charBursts'][i][j][1], vectors_charBurst, vectors_pos, vectors_chunks, df['charBursts'][i][j][5], df['charBursts'][i][j][6], df['charBursts'][i][j][7])  
           
    df["vectors_charBursts"] = big_list_vectors_charBursts
    df["vectors_pos"] = big_list_vectors_pos
    df["vectors_chunks"] = big_list_vectors_chunks
    df["pos_start"] = big_list_pos_start
    df["pos_end"] = big_list_pos_end
    df["frequencies"] = big_list_frequencies
    df["frequencies_in_text"] = big_list_frequencies_in_text
    df["relative_frequencies"] = big_list_relative_frequencies
    df.drop(columns=['charBursts'], inplace=True)

def expand_arrays(df, liste, nom_de_la_colonne):
    df[nom_de_la_colonne] = liste
    expanded = df[nom_de_la_colonne].apply(pd.Series)
    double_expanded = pd.DataFrame()
    for column in expanded.columns:
        expanded_df = expanded[column].apply(pd.Series)
        expanded_df.columns = [f"{nom_de_la_colonne}_{column}_{i}" for i in range(len(expanded_df.columns))]
        double_expanded = pd.concat([double_expanded, expanded_df], axis=1)
    return double_expanded


def expand_columns(df):
    lists_pauses = []
    lists_vectors_charBursts = []
    lists_vectors_pos = []
    lists_vectors_chunks = []
    lists_pos_start = []
    lists_pos_end = []
    lists_frequencies = []
    lists_frequencies_in_text = []
    lists_relative_frequencies = []

    for i in range(len(df)):
        list_pauses = df['pauses'][i].strip("[").strip("]").split(", ")
        lists_pauses.append(list_pauses)

        list_pos_start = df['pos_start'][i]
        lists_pos_start.append(list_pos_start)

        list_pos_end = df['pos_end'][i]
        lists_pos_end.append(list_pos_end)

        list_frequencies = df['frequencies'][i]
        lists_frequencies.append(list_frequencies)

        list_frequencies_in_text = df['frequencies_in_text'][i]
        lists_frequencies_in_text.append(list_frequencies_in_text)

        list_relative_frequencies = df['relative_frequencies'][i]
        lists_relative_frequencies.append(list_relative_frequencies)

        list_vectors_charBursts = df['vectors_charBursts'][i]
        list_vectors_pos = df['vectors_pos'][i]
        list_vectors_chunks = df['vectors_chunks'][i]
        lists_vectors_charBursts.append(list_vectors_charBursts)
        lists_vectors_pos.append(list_vectors_pos)
        lists_vectors_chunks.append(list_vectors_chunks)

    df["pauses"] = lists_pauses
    pauses_expanded = df["pauses"].apply(pd.Series)
    # print(pauses_expanded.shape)
    pauses_expanded.columns = [f"pause_{i}" for i in range(len(pauses_expanded.columns))]

    df["pos_start"] = lists_pos_start
    pos_start_expanded = df["pos_start"].apply(pd.Series)
    pos_start_expanded.columns = [f"pos_start_{i}" for i in range(len(pos_start_expanded.columns))]

    df["pos_end"] = lists_pos_end
    pos_end_expanded = df["pos_end"].apply(pd.Series)
    pos_end_expanded.columns = [f"pos_end_{i}" for i in range(len(pos_end_expanded.columns))]

    df["frequencies"] = lists_frequencies
    frequencies_expanded = df["frequencies"].apply(pd.Series)
    frequencies_expanded.columns = [f"frequency_{i}" for i in range(len(frequencies_expanded.columns))]

    df["frequencies_in_text"] = lists_frequencies_in_text
    frequencies_in_text_expanded = df["frequencies_in_text"].apply(pd.Series)
    frequencies_in_text_expanded.columns = [f"frequency_in_text_{i}" for i in range(len(frequencies_in_text_expanded.columns))]

    df["relative_frequencies"] = lists_relative_frequencies
    relative_frequencies_expanded = df["relative_frequencies"].apply(pd.Series)
    relative_frequencies_expanded.columns = [f"relative_frequency_{i}" for i in range(len(relative_frequencies_expanded.columns))]

    vectors_charBursts_double_expanded = expand_arrays(df, lists_vectors_charBursts, "vectors_charBursts")
    vectors_pos_double_expanded = expand_arrays(df, lists_vectors_pos, "vectors_pos")
    vectors_chunks_double_expanded = expand_arrays(df, lists_vectors_chunks, "vectors_chunk")

    df_expanded = pd.concat([df.drop(columns=["pauses", "pos_start", "pos_end", "frequencies", "frequencies_in_text", "relative_frequencies", "vectors_charBursts", "vectors_pos", "vectors_chunks"]), pauses_expanded, pos_start_expanded, pos_end_expanded, frequencies_expanded, frequencies_in_text_expanded, relative_frequencies_expanded, vectors_charBursts_double_expanded, vectors_pos_double_expanded, vectors_chunks_double_expanded], axis=1)
    return df_expanded

def remove_long_pauses(df):
    pauses_df = df["pauseDur"]
    all_pauses = []
    for index in range(len(pauses_df)):
        line = pauses_df[index]
        all_pauses.append(float(line))
    
    all_pauses_array = np.array(all_pauses)
    q1 = np.percentile(all_pauses_array, 25)
    q3 = np.percentile(all_pauses_array, 75)
    iqr_value = iqr(all_pauses_array)
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    # remove the line in the datafram if pause is longer than 1.5 * iqr_value
    df = df[df["pauseDur"] <= upper_bound]
    df = df.reset_index(drop=True)
    return df

def main():
    people = read_csv(csv_file)
    dataset = []
    for person in people:
        for burst_id in people[person]:
            features = extract_features(people[person][burst_id])
            dataset.append(features)
    df = pd.DataFrame(dataset)
    df = remove_long_pauses(df)

    vectorise_text(df)
    # df.to_csv('vectorised_data.csv')
    df_expanded = expand_columns(df)
    df_expanded.to_pickle('expanded_data.pkl')
    return df_expanded
            
if __name__ == "__main__":
    main()