import collections
import random
from ultis.Tokenizer import Tokenizer
import torch
from utils import special_char


class DataBuilder:
    def __init__(
        self,
        rng,
        padding_id,
        mask_token,
        seq_max_length,
        word_max_length,
        custom_tokenizer: Tokenizer,
        char_tokenizer: Tokenizer,
    ):
        self.rng = rng
        self.custom_tokenizer = custom_tokenizer
        self.char_tokenizer = char_tokenizer
        self.word_max_length = word_max_length
        self.seq_max_length = seq_max_length
        self.padding_id = padding_id
        self.mask_token = mask_token

    def build_data(self, inputs):
        word_input_ids = []
        mask_token_positions = []
        target = []
        char_input_ids = []
        for seq in inputs:
            seq_char_ids = []
            seq_char = seq
            for word in seq_char.split(" "):
                for start_char in special_char.keys():
                    if word.startswith(start_char):
                        seq_char = seq_char.replace(
                            word, special_char[start_char] + word
                        )
            tokens = self.char_tokenizer.tokenize(seq_char)
            for word in tokens:
                word_char_ids = self.char_tokenizer.convert_tokens_to_ids(word)
                word_char_ids = padding(
                    word_char_ids, self.word_max_length, self.padding_id
                )
                seq_char_ids.append(word_char_ids)
            char_pad = padding([], self.word_max_length, self.padding_id)
            seq_char_ids = padding(seq_char_ids, self.seq_max_length, char_pad)
            tokens = self.custom_tokenizer.tokenize(seq)
            if len(tokens) > self.seq_max_length:
                for i in range(len(tokens) - self.seq_max_length):
                    list_tokens = tokens[i : i + self.seq_max_length]
                    token_masked, masked_positions, masked_label = (
                        create_masked_lm_predictions(
                            rng=self.rng,
                            tokens=list_tokens,
                            vocab_words=self.custom_tokenizer.inv_vocab,
                            mask_token=self.mask_token,
                            masked_lm_prob=0.15,
                            max_predictions_per_seq=5,
                        )
                    )
                    word_input_ids = word_input_ids + [
                        self.custom_tokenizer.convert_tokens_to_ids(_)
                        for _ in token_masked
                    ]
                    for _ in masked_positions:
                        __ = torch.zeros(self.seq_max_length, dtype=bool)
                        __[_] = True
                        mask_token_positions.append(__)
                    target = target + [
                        self.custom_tokenizer.convert_tokens_to_ids(_)
                        for _ in masked_label
                    ]
                    char_input_ids = char_input_ids + [
                        seq_char_ids for _ in range(len(token_masked))
                    ]
            else:
                token_masked, masked_positions, masked_label = (
                    create_masked_lm_predictions(
                        rng=self.rng,
                        tokens=tokens,
                        vocab_words=self.custom_tokenizer.inv_vocab,
                        mask_token=self.mask_token,
                        masked_lm_prob=0.15,
                        max_predictions_per_seq=5,
                    )
                )
                word_input_ids = word_input_ids + [
                    padding(
                        self.custom_tokenizer.convert_tokens_to_ids(_),
                        max_length=self.seq_max_length,
                        padding_id=1,
                    )
                    for _ in token_masked
                ]
                for _ in masked_positions:
                    __ = torch.zeros(self.seq_max_length, dtype=bool)
                    __[_] = True
                    mask_token_positions.append(__)
                target = target + [
                    self.custom_tokenizer.convert_tokens_to_ids(_) for _ in masked_label
                ]
                char_input_ids = char_input_ids + [
                    seq_char_ids for _ in range(len(token_masked))
                ]

        return (
            torch.tensor(char_input_ids),
            torch.tensor(word_input_ids),
            torch.stack(mask_token_positions),
            target,
        )


def padding(input_ids, max_length, padding_id):
    len_sentence = len(input_ids)
    while len(input_ids) > max_length:
        del input_ids[-1]
    for i in range(max_length - len_sentence):
        input_ids.append(padding_id)
    return input_ids


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_char_input(
    tokens: list, vocab_chars, mask_token, rng: random.Random, masked_lm_prob=0.25
):
    size_alter = int(len(tokens) * (1 + masked_lm_prob))
    number_range = list(range(size_alter))
    output = tokens
    cand_indexes = rng.sample(number_range, size_alter - len(output))
    cand_indexes.sort()
    for index in cand_indexes:
        random_char_token = vocab_chars[rng.randint(0, len(vocab_chars) - 1)]
        output.insert(index, random_char_token)
    output.append(mask_token)
    return output


def create_masked_lm_predictions(
    tokens: list,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words: dict,
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
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = []

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )
    list_masked_lms = []
    list_masked_lm_positions = []
    list_masked_lm_labels = []
    covered_indexes = set()
    while len(covered_indexes) < len(cand_indexes):
        masked_lms = []
        tokens_masked = list(tokens)
        for index in cand_indexes:
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = mask_token
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            tokens_masked[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            if len(masked_lms) >= num_to_predict:
                masked_lms = sorted(masked_lms, key=lambda x: x.index)
                masked_lm_positions = []
                masked_lm_labels = []
                for p in masked_lms:
                    masked_lm_positions.append(p.index)
                    masked_lm_labels.append(p.label)
                break
            if len(covered_indexes) == len(cand_indexes):
                masked_lms = sorted(masked_lms, key=lambda x: x.index)
                masked_lm_positions = []
                masked_lm_labels = []
                for p in masked_lms:
                    masked_lm_positions.append(p.index)
                    masked_lm_labels.append(p.label)
        output_tokens.append(tokens_masked)
        list_masked_lms.append(masked_lms)
        list_masked_lm_positions.append(masked_lm_positions)
        list_masked_lm_labels.append(masked_lm_labels)

    return (output_tokens, list_masked_lm_positions, list_masked_lm_labels)
