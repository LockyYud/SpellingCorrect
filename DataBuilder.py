import collections
import unicodedata
from transformers import AutoTokenizer
import torch


class DataBuilderForElectra:
    def __init__(
        self,
        rng,
        seq_max_length,
        tokenizer,
    ):
        self.rng = rng
        self.tokenizer: AutoTokenizer = tokenizer
        self.seq_max_length = seq_max_length

    def build_data(self, inputs):
        input_original = []
        input_masked = []
        for sentence in inputs:
            tokens = split_token(sentence)
            tokens.insert(0, self.tokenizer.cls_token)
            if len(tokens) > self.seq_max_length:
                tokens = tokens[: self.seq_max_length]
            tokens_masked = create_masked_lm_predictions(
                tokens=tokens,
                masked_lm_prob=0.20,
                max_predictions_per_seq=3,
                rng=self.rng,
            )
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)
            input_original.append(
                padding(
                    tokens, self.seq_max_length, padding_id=self.tokenizer.pad_token_id
                )
            )
            input_masked.append(
                padding(
                    tokens_masked,
                    self.seq_max_length,
                    padding_id=self.tokenizer.pad_token_id,
                )
            )
        return torch.Tensor(input_original).long(), torch.Tensor(input_masked).long()


def split_token(text):
    text = _clean_text(text)
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
        token = token.lower()
        split_tokens.extend(run_split_on_punc(token))
    return [token for token in whitespace_tokenize(" ".join(split_tokens))]


def padding(input_ids, max_length, padding_id):
    len_sentence = len(input_ids)
    while len(input_ids) > max_length:
        del input_ids[-1]
    for i in range(max_length - len_sentence):
        input_ids.append(padding_id)
    return input_ids


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(
    tokens,
    masked_lm_prob,
    max_predictions_per_seq,
    rng,
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for i, token in enumerate(tokens):
        if token == cls_token or token == sep_token:
            continue
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            masked_token = mask_token
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


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


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


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
