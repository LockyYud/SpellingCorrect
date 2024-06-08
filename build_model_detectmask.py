import torch
import torch.nn as nn
import utils
import funtion
from SCModel import ModelSC
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, AutoTokenizer
from CustomDataset import CustomDataset
from data_builder import DataBuilder
from Tokenizer import Tokenizer
import os
import time
import random

batch_size = 150

os.makedirs("data", exist_ok=True)
## Define model and tokenizer phobert
phobert = RobertaForMaskedLM.from_pretrained("vinai/phobert-base-v2")
pho_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
sentences = funtion.load_data_and_vocab(
    path_file_data="data/sentences.txt",
    path_file_vocab_char="data/char_vocab.txt",
    path_file_vocab_word="data/word_vocab.txt",
    special_token=["<unk>", "<pad>", "<mask>", "<cls>", "<sep>"],
    size_data=40000,
)

## define Tokenizer
custom_tokenizer = Tokenizer(vocab_file="data/word_vocab.txt", unk_token="<unk>")
char_tokenizer = Tokenizer(
    vocab_file="data/char_vocab.txt", unk_token="<unk>", char_level=True
)
data_builder = DataBuilder(
    rng=random.Random(42),
    padding_id=pho_tokenizer.pad_token_id,
    mask_token=pho_tokenizer.mask_token,
    seq_max_length=64,
    word_max_length=10,
    bert_tokenizer=pho_tokenizer,
    custom_tokenizer=custom_tokenizer,
    char_tokenizer=char_tokenizer,
)
## Build data
char_input_ids, phobert_word_ids, mask_token_positions, target = (
    data_builder.build_data(inputs=sentences)
)
## define dataset

data_train = CustomDataset(
    char_input_ids=char_input_ids,
    phobert_word_ids=phobert_word_ids,
    mask_token_positions=mask_token_positions,
    target=target,
)
train_dataloader = DataLoader(
    dataset=data_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_train.collate_fn,
)


def train_loop(dataloader, model, loss_fn, optimizer, device, phobert):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, data in enumerate(dataloader):
        # Define data
        input_ids = data["input_ids"]
        batch_size, seq_len = input_ids.shape
        target = data["target"]
        # Generator
        attn_mask = input_ids != tokenizer.pad_token_id
        with torch.no_grad():
            phobert.eval()
            input_replaced = torch.argmax(phobert(input_ids, attn_mask).logits, dim=-1)
        # Discriminator
        key_mask_padding = input_ids == tokenizer.pad_token_id
        input_replaced[key_mask_padding] = tokenizer.pad_token_id
        pred = model(input_replaced)
        # Compute loss
        target = target[attn_mask]
        pred = pred[pred]
        pred = pred.view(batch_size * seq_len, 2)
        target = target.view(-1)
        loss = loss_fn(pred, target)
        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


learning_rate = 1e-3
batch_size = 100
epochs = 1
optimizer = torch.optim.Adam(modelElectraSC.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(
        dataloader=train_dataloader,
        model=modelElectraSC,
        loss_fn=loss_fn,
        optimizer=optimizer,
        phobert=phobert,
        device=device,
    )
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")
