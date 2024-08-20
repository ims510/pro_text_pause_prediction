import csv
import pandas as pd

csv_file = "/Users/madalina/Documents/M1TAL/stage_GC/pro_text/predictions/planification.csv"
# csv_file = "/Users/madalina/Documents/M1TAL/stage_GC/pro_text/predictions/production_p+s1.csv"

def process_row(row, data):
    if row[3] not in data:
        data[row[3]] = [row[i] for i in range(4, 15)]
        data[row[3]].append(row[17])
        # data[row[3]].append(row[18]) # this is the type of burst (P, R, ER) 
        char_burst = [(row[15], row[16], row[19])]
        data[row[3]].append(char_burst)
    else:
        char_burst = (row[15], row[16], row[19])
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
    
    charBursts = burst_data[12]  # This is the list of tuples

    # Feature extraction from charBurst
    avg_shift = sum(abs(int(end) - int(start)) for start, end, _ in charBursts) / len(charBursts) # not sure if good metric
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
        'avg_shift': avg_shift,
        'num_actions': num_actions,
        'pauseDur': pauseDur # This is the target we want to predict
    }

def main():
    people = read_csv(csv_file)
    dataset = []
    for person in people:
        for burst_id in people[person]:
            features = extract_features(people[person][burst_id])
            dataset.append(features)
    df = pd.DataFrame(dataset)
    return df
            
if __name__ == "__main__":
    main()
