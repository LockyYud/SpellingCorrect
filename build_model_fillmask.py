import torch
import torch.nn as nn
import utils
import ultis.funtion as funtion
from model.SCModel import ModelSC
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, AutoTokenizer
from ultis.CustomDataset import CustomDataset
from ultis.DataBuilder import DataBuilder
from ultis.Tokenizer import Tokenizer
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
## check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ModelSC(
    character_level_d_model=256,
    word_level_d_model=768,
    num_heads_char_encoder=4,
    num_layers_char_encoder=4,
    num_heads_total=4,
    num_layers_total=4,
    dff=128,
    encoder_total=phobert.roberta.encoder.layer[3:6],
    character_vocab_size=len(char_tokenizer.vocab),
    word_vocab_size=len(custom_tokenizer.vocab),
    bert_embedding_layer=phobert.roberta.embeddings,
    bert_encoder_layers=phobert.roberta.encoder.layer[:3],
    device=device,
)


def train_loop(dataloader, model: ModelSC, loss_fn, optimizer, device, phobert):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        phobert_word_ids_train = X["phobert_word_ids"].to(device)
        char_input_ids_train = X["char_input_ids"].to(device)
        mask_token_positions_train = X["mask_token_positions"].to(device)
        pred = model(
            char_input_ids_train, phobert_word_ids_train, mask_token_positions_train
        )
        target = (
            torch.Tensor([item for sublist in y for item in sublist]).long().to(device)
        )
        loss = loss_fn(pred, target)
        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


learning_rate = 1e-3
epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
model.to(device)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(
        dataloader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        phobert=phobert,
        device=device,
    )
print("Done!")
