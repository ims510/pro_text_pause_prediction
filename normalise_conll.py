import stanza
import os

nlp = stanza.Pipeline('fr', processors='tokenize, pos, lemma, depparse')


def create_new_conll(data, output_file):
    previous_text_version = 0
    previous_sent_id = 0
    with open (output_file, "w", encoding="utf-8") as f:
        misc =""
        for i in range(len(data)):
            elements = [str(element) for element in data[i]]

            concatenated_elements = "\n".join(elements)
            data[i] = concatenated_elements
            conll_sent = data[i].split("\n")

            text_version = conll_sent [0] #literally: # text_version=0
            text_version = int(text_version.split("=")[1])

            sentence_id = conll_sent[1] #literally: # sentence_id=1
            sent_id = 1

            if text_version == previous_text_version and sent_id <= previous_sent_id:
                sent_id = previous_sent_id + 1
            sentence_complete = conll_sent[2:]
            sentence = ""
            original_tokens = []
            for token in sentence_complete:

                elements = token.split("\t")
                item_dict = {'first_elements': tuple(elements[:10]), 'misc': elements[9]} if len(elements) > 1 else {'first_elements': tuple(elements), 'misc': ""}
                original_tokens.append(item_dict)

                if len(elements) > 1:
                    sentence += elements[1] + " "

            doc = nlp(sentence)
            for j in range(len(doc.sentences)):
                mini_sentence = ""
                mini_sentence = " ".join([token.text for token in doc.sentences[j].words])
                if j > 0:
                    f.write("# text_version=")
                    f.write(str(text_version))
                    previous_text_version = text_version
                    f.write("\n")

                    f.write("# sentence_id=")
                    f.write(str(sent_id))
                    previous_sent_id = sent_id
                    sent_id += 1
                    f.write("\n")

                    f.write(f"# text = {mini_sentence}")
                    f.write("\n")
                else:
                    f.write("# text_version=")
                    f.write(str(text_version))
                    previous_text_version = text_version
                    f.write("\n")

                    f.write("# sentence_id=")
                    f.write(str(sent_id))
                    previous_sent_id = sent_id
                    sent_id += 1
                    f.write("\n")

                    f.write(f"# text = {mini_sentence}")
                    f.write("\n")
                for i in range(len(doc.sentences[j].words)):
                    token = doc.sentences[j].words[i]

                    xpos = token.xpos if not "None" else "_"
                    deps = token.deps if not "None" else "_"
                    for element in original_tokens:
                        if element['first_elements'][1] == token.text and int(element['first_elements'][0]) == token.id:
                            misc = element['misc']
                            break   
                    f.write(f"{token.id}\t{token.text}\t{token.lemma}\t{token.pos}\t{xpos}\t{token.feats}\t{token.head}\t{token.deprel}\t{deps}\t{misc}\n")
                f.write("\n")

    print(f"File {output_file} created.")

def extract_data(file):
    data_file = open(file, "r")
    data = []

    # Read file and split into sentences as lists of lines
    current_sentence = []
    for line in data_file:
        if line == "\n":
            if current_sentence:  # Ensure the sentence is not empty
                data.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append(line.strip())  # Remove newline characters

    if current_sentence:  # Add the last sentence if file doesn't end with a newline
        data.append(current_sentence)

    data_file.close()  # Close the file after reading

    last_text_version = None

    # Insert last_text_version where needed
    for i in range(len(data)):
        if not data[i][0].startswith("# text_version"):
            if last_text_version is not None:
                data[i].insert(0, last_text_version)
        else:
            last_text_version = data[i][0]


    for i in range(len(data)):
        expected_number = 1
        for j in range(2, len(data[i])):  # Assuming the first two lines are comments
            components = data[i][j].split('\t')
            # print(components)
            if len(components) > 1:  # Ensure the line is not a comment
                components[0] = str(expected_number)
                data[i][j] = '\t'.join(components)
                expected_number += 1

    return data



def main():
    source_dir = "/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/conll"
    target_dir = "/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/conll_clean"
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterate through all files in the source directory
    for file in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, file)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(source_file_path):
            # Construct the target file path
            target_file_name = os.path.splitext(file)[0] + "_clean" + os.path.splitext(file)[1]
            target_file_path = os.path.join(target_dir, target_file_name)
            
            # Process the file
            data = extract_data(source_file_path)
            create_new_conll(data, target_file_path)


main()