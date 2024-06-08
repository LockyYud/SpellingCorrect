from CreateVocab import CreateVocab
from utils import write_list_data_to_file
import pandas as pd


def load_data_and_vocab(
    path_file_vocab_char=None,
    path_file_vocab_word=None,
    path_file_data=None,
    size_data=10000,
    special_token=None,
    vocab_size=1000,
):
    file = open(path_file_data, "r")
    sentences = file.readlines()
    sentences = [sentence.replace("\n", "").lower() for sentence in sentences]
    createVocab = CreateVocab(special_token=special_token)
    if path_file_vocab_word is not None:
        word_vocab = createVocab.create_word_vocab(sentences, vocab_size=vocab_size)
        write_list_data_to_file(path_file_vocab_word, word_vocab)
    if path_file_vocab_char is not None:
        char_vocab = createVocab.create_character_vocab(sentences)
        write_list_data_to_file(path_file_vocab_char, char_vocab)
    return sentences
