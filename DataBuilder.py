import collections
import string
import torch

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class DataBuilderForElectra:
    def __init__(
        self, mask_token, padding_token, sep_token, cls_token, eos_token, max_len, rng
    ):
        self.mask_token = mask_token
        self.padding_token = padding_token
        self.cls_token = cls_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.max_len = max_len
        self.rng = rng

    def build_data_for_generator(self, input):
        input_masked = []
        token_masked_positions = []
        token_masked_labels = []
        for s in input:
            output_tokens, masked_lm_positions, masked_lm_labels = (
                self.create_masked_lm_predictions(
                    self.replace_punctuation(s).split(), 0.15, 3, self.rng
                )
            )
            input_masked.append(
                self.padding(
                    output_tokens,
                    max_length=self.max_len,
                    padding_id=self.padding_token,
                )
            )
            token_masked_positions.append(masked_lm_positions)
            token_masked_labels.append(masked_lm_labels)
        return input_masked, token_masked_positions, token_masked_labels

    def build_target(self, input):
        target = [
            self.padding(
                self.replace_punctuation(s).split(),
                max_length=self.max_len,
                padding_id=self.padding_token,
            )
            for s in input
        ]
        return target

    # def build_target
    def create_masked_lm_predictions(
        self, tokens, masked_lm_prob, max_predictions_per_seq, rng
    ):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == self.cls_token or token == self.sep_token:
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
                # 80% of the time, replace with [MASK]
                masked_token = self.mask_token

                output_tokens[index] = masked_token

                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return (output_tokens, masked_lm_positions, masked_lm_labels)

    def replace_punctuation(self, text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, "").lower().strip()
        return text

    def padding(self, input_ids: list, max_length, padding_id):
        input_ids.insert(0, self.cls_token)
        if len(input_ids) < max_length:
            input_ids.append(self.eos_token)
        len_sentence = len(input_ids)
        while len(input_ids) > max_length:
            del input_ids[-1]
        for i in range(max_length - len_sentence):
            input_ids.append(padding_id)
        return input_ids
