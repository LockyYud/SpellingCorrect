from torch.nn.utils.rnn import pad_sequence
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, char_input_ids, phobert_word_ids, mask_token_positions, target):
        self.char_input_ids = char_input_ids
        self.phobert_word_ids = phobert_word_ids
        self.mask_token_positions = mask_token_positions
        self.target = target

    def __getitem__(self, idx):
        target = self.target[idx]
        phobert_word_ids = self.phobert_word_ids[idx]
        mask_token_positions = self.mask_token_positions[idx]
        char_input_ids = self.char_input_ids[idx]

        return {
            "phobert_word_ids": phobert_word_ids,
            "mask_token_positions": mask_token_positions,
            "char_input_ids": char_input_ids,
        }, target

    def __len__(self):
        return len(self.target)

    def collate_fn(self, batch):
        phobert_word_ids = [item[0]["phobert_word_ids"] for item in batch]
        mask_token_positions = [item[0]["mask_token_positions"] for item in batch]
        char_input_ids = [item[0]["char_input_ids"] for item in batch]
        targets = [item[1] for item in batch]

        # Pad sequences
        phobert_word_ids_padded = pad_sequence(
            phobert_word_ids, batch_first=True, padding_value=0
        )
        mask_token_positions_padded = pad_sequence(
            mask_token_positions, batch_first=True, padding_value=0
        )
        char_input_ids_padded = pad_sequence(
            char_input_ids, batch_first=True, padding_value=0
        )

        return {
            "phobert_word_ids": phobert_word_ids_padded,
            "mask_token_positions": mask_token_positions_padded,
            "char_input_ids": char_input_ids_padded,
        }, targets
