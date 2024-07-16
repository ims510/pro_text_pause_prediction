from dataclasses import dataclass, field
from typing import Self

@dataclass
class Character:
    name: str
    id: int
    status: bool
    pause_before: int
    end_of_word: bool = False
    end_of_sentence: bool = False

    @classmethod
    def from_string(cls, string: str) -> Self:
        fields = string.split("__")
        char_id_list = [field.split('=')[1] for field in fields if field.startswith('charID')]
        pause_value_list = [field.split('=')[1] for field in fields if field.startswith('pause')]
        char_status = [field.split('=')[1] for field in fields if field.startswith('charStatus')]
        char = [field[-1] for field in fields if field.startswith('char=')]
        if char_id_list and pause_value_list and char_status and char:
            char_id = int(char_id_list[0])
            pause_value = pause_value_list[0]
            pause_value = 0 if pause_value == 'na' else int(pause_value)
            char_status = True if char_status[0] == "True" else False
            char = char[0]
            return cls(char, char_id, char_status, pause_value, False, False)
        else:
            raise ValueError("Not enough fields in character details")

@dataclass
class Token:
    id: int
    text: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str
    chars: list[Character]

    @classmethod
    def from_string(cls, string: str) -> Self:
        fields = string.split("\t")
        if len(fields) < 10:
            raise ValueError("Not enough fields in CoNLL-U line")
        id = int(fields[0]) 
        text = fields[1] 
        lemma = fields[2] 
        upos = fields[3] 
        xpos = fields[4] 
        feats = fields[5] 
        head = int(fields[6]) 
        deprel = fields[7] 
        deps = fields[8] 
        misc = fields[9] 
        chars = []
        char_details = misc.split("|")
        for detail in char_details:
            char = Character.from_string(detail)
            chars.append(char)
        chars[-1].end_of_word = True
        return cls(id, text, lemma, upos, xpos, feats, head, deprel, deps, misc, chars)


@dataclass
class Sentence:
    text_version: int
    sentence_id: int
    text: str
    tokens: list[Token]

    @classmethod
    def from_string(cls, string:str) -> Self:
        lines = string.split("\n")
        if lines[0].startswith("# text_version"):
            text_version = int(lines[0].split("=")[1])
        else:
            raise ValueError("No # text_version in sentence")
        if lines[1].startswith("# sentence_id"):
            sentence_id = int(lines[1].split("=")[1])
        else:
            raise ValueError("No # sentence_id in sentence")
        if lines[2].startswith("# text"):
            text = lines[2].split("=")[1].strip()
        else:
            raise ValueError("No # text in sentence")
        tokens = []
        for line in lines[3:]:
            token = Token.from_string(line)
            tokens.append(token)
        tokens[-1].chars[-1].end_of_sentence = True
        return cls(text_version, sentence_id, text, tokens)
