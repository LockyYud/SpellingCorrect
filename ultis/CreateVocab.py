import collections
import re
import unicodedata
import six
from utils import vietnamese_characters, special_token_sign_vietnamese


class CreateVocab(object):
    def __init__(self, special_token):
        self.special_token = special_token

    def create_word_vocab(self, data_text, vocab_size=None):
        print("Creating vocab:")
        word_vocab = collections.OrderedDict()
        list_word_token = collections.OrderedDict()
        for sentence in data_text:
            list_token = convert_sentence_to_token(sentence)
            for token in list_token:
                if contains_number(token) or contains_special_character(token):
                    continue
                if token in list_word_token:
                    list_word_token[token] = list_word_token[token] + 1
                else:
                    list_word_token[token] = 1
        list_word_token = collections.OrderedDict(
            sorted(list_word_token.items(), key=lambda x: x[1], reverse=True)
        )
        index = 0
        for token in self.special_token:
            word_vocab[token] = index
            index += 1
        for token in list_word_token:
            word_vocab[token] = index
            if vocab_size != None:
                if index > vocab_size - len(self.special_token):
                    break
            index += 1
        return word_vocab

    def create_character_vocab(self, file_text, vocab_size=None):
        character_vocab = collections.OrderedDict()
        list_char_token = collections.OrderedDict()
        for senetence in file_text:
            for char in senetence:
                if contains_special_character(token=char):
                    continue
                if char in list_char_token:
                    list_char_token[char] = list_char_token[char] + 1
                else:
                    list_char_token[char] = 1
        list_char_token = collections.OrderedDict(
            sorted(list_char_token.items(), key=lambda x: x[1], reverse=True)
        )
        index = 0
        for token in self.special_token:
            character_vocab[token] = index
            index += 1
        for token in list_char_token:
            if token in vietnamese_characters:
                continue
            character_vocab[token] = index
            if vocab_size != None:
                if index > vocab_size - len(self.special_token):
                    break
            index += 1
        for token in special_token_sign_vietnamese.values():
            character_vocab[token] = index
            index += 1
        return character_vocab


def contains_special_character(token):
    for char in token:
        if _is_whitespace(char=char):
            return True
        # if _is_punctuation(char=char):
        #     return True
    return False


def contains_number(text):
    pattern = re.compile(r"[0-9]")
    return bool(pattern.search(text))


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def convert_sentence_to_token(text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
        # if self.do_lower_case:
        token = token.lower()
        # token = self._run_strip_accents(token)
        split_tokens.extend(run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))


def run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False
