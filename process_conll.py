from data_structure import Sentence, Token, Character
import os

def read_conll(source_dir):
    sentences = []
    for file in os.listdir(source_dir):
        if file.startswith("."):
            continue
        source_file_path = os.path.join(source_dir, file)
        data_file = open(source_file_path, "r")
        # print("Processing file: ", source_file_path)
        sentence_string = ""
        data = []
        for line in data_file:
            if line[0] == "\n":
                data.append(sentence_string)
                sentence_string = ""
            else:
                sentence_string = sentence_string + line
        for sentence_string in data:
            sentence_string = sentence_string.strip()
            if sentence_string != "":
                sentence = Sentence.from_string(sentence_string)
                sentences.append(sentence)
        data_file.close()
    return sentences

sentences = read_conll("/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/conll_clean")
