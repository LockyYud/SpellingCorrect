from ultis.CreateVocab import CreateVocab
from utils import write_list_data_to_file
import pandas as pd
import string


def load_data_and_vocab(
    path_file_data=None,
    data_size=1000,
):
    file = open(path_file_data, "r")
    sentences = file.readlines()
    sentences = [
        replace_punctuation(sentence.replace("\n", "").lower())
        for sentence in sentences
    ]
    return sentences[:data_size]


def replace_punctuation(text, replacement=""):
    translator = str.maketrans({key: replacement for key in string.punctuation})
    return text.translate(translator)
