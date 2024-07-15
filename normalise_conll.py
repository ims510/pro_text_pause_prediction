import stanza
import os

nlp = stanza.Pipeline('fr', processors='tokenize, pos, lemma, depparse')

def get_token_text_from_misc(misc):
    misc = misc.split("|")
    word = ""
    for char_details in misc:
        char_details = char_details.split("__")
        char = char_details[1].split("=")[1]
        word += char
    return word

def merge_elements(original_list, stanza_list):
    new_list = []
    min_len = min(len(original_list), len(stanza_list))
    if min_len == len(original_list):
        for i in range(min_len):
            orig_misc = original_list[i][9]
            stanza_misc = stanza_list[i][9]
            if orig_misc == stanza_misc:
                new_list.append(stanza_list[i])

            else:
                new_element = [stanza_list[i][0], original_list[i][1], stanza_list[i][2], stanza_list[i][3], stanza_list[i][4], stanza_list[i][5], stanza_list[i][6], stanza_list[i][7], stanza_list[i][8], orig_misc]
                
                new_list.append(new_element)
    else:
        for i in range(min_len):
            orig_misc = original_list[i][9]
            stanza_misc = stanza_list[i][9]
            if orig_misc == stanza_misc:
                new_list.append(stanza_list[i])

            else:
                new_element_1 = [int(original_list[i][0]), original_list[i][1], original_list[i][2], stanza_list[i][3], stanza_list[i][4], stanza_list[i][5], stanza_list[i][6], stanza_list[i][7], stanza_list[i][8], orig_misc]
                new_element_2 = [int(original_list[i+1][0]), original_list[i+1][1], original_list[i+1][2], stanza_list[i][3], stanza_list[i][4], stanza_list[i][5], stanza_list[i][6], stanza_list[i][7], original_list[i+1][8], original_list[i+1][9]]
                new_list.append(new_element_1)
                new_list.append(new_element_2)
    return new_list

def create_new_conll(data, output_file):
    previous_text_version = 0
    previous_sent_id = 0
    with open (output_file, "w", encoding="utf-8") as f:
        misc =""
        for i in range(len(data)):
            elements = [str(element) for element in data[i]]
            text_version = int(elements[0].split("=")[1]) #literally: # text_version=0
            sent_id = 1

            if text_version == previous_text_version and sent_id <= previous_sent_id:
                sent_id = previous_sent_id + 1

            sentence_complete = elements[2:]

            sentence = ""
            original_tokens = []
            list_elements = []
            for token in sentence_complete:

                conll_columns = token.split("\t")
                list_elements.append(conll_columns)
                item_dict = {'first_elements': tuple(conll_columns[:10]), 'misc': conll_columns[9]} if len(conll_columns) > 1 else {'first_elements': tuple(conll_columns), 'misc': ""}
                original_tokens.append(item_dict)
                
                if len(conll_columns) > 1:
                    sentence += conll_columns[1] + " "


            doc = nlp(sentence)
            stanza_nb_tokens = 0
            for j in range(len(doc.sentences)):
                stanza_nb_tokens += len(doc.sentences[j].words)
            list_stanza_elements = []
            if stanza_nb_tokens != len(list_elements):
                for j in range(len(doc.sentences)):
                    for i in range(len(doc.sentences[j].words)):
                        token = doc.sentences[j].words[i]
                        
                        xpos = token.xpos if not "None" else "_"
                        deps = token.deps if not "None" else "_"
                        if j == 0:
                            id_to_check_against = token.id
                        else:
                            id_to_check_against = token.id + len(doc.sentences[j-1].words)
                        for element in original_tokens:
                            if element['first_elements'][1] == token.text and int(element['first_elements'][0]) == id_to_check_against:
                                misc = element['misc']
                                break
                        list_stanza_elements.append([token.id, token.text, token.lemma, token.pos, xpos, token.feats, token.head, token.deprel, deps, misc])
                new_list = merge_elements(list_elements, list_stanza_elements)
                iterations = 0
                for element in new_list:
                    if element[0] == 1:
                        mini_sentence = " ".join([token.text for token in doc.sentences[iterations].words])
                        if iterations > 0:
                            f.write("\n")
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

                        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\t{element[4]}\t{element[5]}\t{element[6]}\t{element[7]}\t{element[8]}\t{element[9]}\n")
                        iterations += 1
                    else:
                        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\t{element[4]}\t{element[5]}\t{element[6]}\t{element[7]}\t{element[8]}\t{element[9]}\n")
            else:
                nb_words_in_sentence = []
                for j in range(len(doc.sentences)):
                    nb_words_in_sentence.append(len(doc.sentences[j].words))
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
                        if j == 0:
                            id_to_check_against = token.id
                        else:
                            id_to_check_against = token.id + sum(nb_words_in_sentence[:j])                    

                        for element in original_tokens:

                            if element['first_elements'][1] == token.text and int(element['first_elements'][0]) == id_to_check_against:
                                misc = element['misc']
                                break   
                        f.write(f"{token.id}\t{token.text}\t{token.lemma}\t{token.pos}\t{xpos}\t{token.feats}\t{token.head}\t{token.deprel}\t{deps}\t{misc}\n")
                    f.write("\n")

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