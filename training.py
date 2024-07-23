import torch
import torch.nn as nn
import time
import random
import os
import utils
import ultis.funtion as funtion
from model.SCModel import ModelSC
from model.DetectModel import DetectModel
from torch.utils.data import DataLoader
from ultis.CustomDataset import CustomDataset
from ultis.DataBuilder import DataBuilder
from ultis.Tokenizer import Tokenizer

batch_size = 200
word_embed_dim = 128
char_embed_dim = 128
device = torch.device("cuda" if torch.cuda else "cpu")

sentences = funtion.load_data_and_vocab(
    path_file_data="data/sentences.txt",
    data_size=500,
)
custom_tokenizer = Tokenizer(vocab_file="vocab.txt", unk_token="<unk>")
char_tokenizer = Tokenizer(
    vocab_file="data/char_vocab.txt", unk_token="<unk>", char_level=True
)
data_builder = DataBuilder(
    rng=random.Random(42),
    padding_id=1,
    mask_token="<mask>",
    seq_max_length=64,
    word_max_length=10,
    custom_tokenizer=custom_tokenizer,
    char_tokenizer=char_tokenizer,
)
## Build data
char_input_ids, word_input_ids, mask_token_positions, target = data_builder.build_data(
    inputs=sentences
)
data_train = CustomDataset(
    char_input_ids=char_input_ids,
    word_input_ids=word_input_ids,
    mask_token_positions=mask_token_positions,
    target=target,
)
train_dataloader = DataLoader(
    dataset=data_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_train.collate_fn,
)

model_detect = DetectModel(
    embed_dim=word_embed_dim,
    num_heads=4,
    num_layer=4,
    vocab_size=len(custom_tokenizer.vocab),
    device=device,
)

model_fill = ModelSC(
    character_level_d_model=char_embed_dim,
    word_level_d_model=word_embed_dim,
    num_heads_char_encoder=4,
    num_heads_word_encoder=4,
    num_layers_char_encoder=4,
    num_layers_word_encoder=4,
    dff=128,
    character_vocab_size=len(char_tokenizer.vocab),
    word_vocab_size=len(custom_tokenizer.vocab),
    device=device,
)


def train_loop(
    dataloader,
    model_detect: DetectModel,
    model_fill: ModelSC,
    loss_fn_model_detect,
    loss_fn_model_fill,
    optimizer_model_detect,
    optimizer_model_fill,
    device,
):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model_detect.train()
    model_fill.train()
    for batch, (X, y) in enumerate(dataloader):
        # Data for fill
        word_input_ids_fill: torch.Tensor = X["word_input_ids"].to(device)
        char_input_ids_fill = X["char_input_ids"].to(device)
        mask_token_positions_fill = X["mask_token_positions"].to(device)

        # Train model fill
        pred_fill: torch.Tensor = model_fill(
            char_input_ids_fill, word_input_ids_fill, mask_token_positions_fill
        )
        target_fill = (
            torch.Tensor([item for sublist in y for item in sublist]).long().to(device)
        )
        loss_fill = loss_fn_model_fill(pred_fill, target_fill)
        loss_fill.backward(retain_graph=True)
        optimizer_model_fill.step()
        optimizer_model_fill.zero_grad()

        # Data for detect
        input_ids_detect = word_input_ids_fill.clone()
        input_ids_detect[mask_token_positions_fill] = pred_fill.argmax(-1)
        word_input_ids_fill[mask_token_positions_fill] = target_fill
        target_detect = (word_input_ids_fill != input_ids_detect).to(device).long()
        target_detect = target_detect.view(
            target_detect.shape[0] * target_detect.shape[1]
        )
        # Train model detect
        pred_detect: torch.Tensor = model_detect(input_ids_detect)
        pred_detect = pred_detect.view(
            pred_detect.shape[0] * pred_detect.shape[1], pred_detect.shape[-1]
        )
        loss_detect = loss_fn_model_detect(pred_detect, target_detect)
        loss_detect.backward()
        optimizer_model_detect.step()
        optimizer_model_detect.zero_grad()
        if batch % 100 == 0:
            loss_fill, current_fill = loss_fill.item(), batch * batch_size
            loss_detect, current_detect = loss_detect.item(), batch * batch_size
            print(
                f"loss: {loss_fill:>7f}  [{current_fill:>5d}/{size:>5d}]",
                f"loss: {loss_detect:>7f}  [{current_detect:>5d}/{size:>5d}]",
            )


def train_loop_fill_masked(
    dataloader,
    model_fill: ModelSC,
    loss_fn_model_fill,
    optimizer_model_fill,
    device,
):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model_fill.train()
    for batch, (X, y) in enumerate(dataloader):
        word_input_ids_fill: torch.Tensor = X["word_input_ids"].to(device)
        char_input_ids_fill = X["char_input_ids"].to(device)
        mask_token_positions_fill = X["mask_token_positions"].to(device)

        # Train model fill
        pred_fill: torch.Tensor = model_fill(
            char_input_ids_fill, word_input_ids_fill, mask_token_positions_fill
        )
        target_fill = (
            torch.Tensor([item for sublist in y for item in sublist]).long().to(device)
        )
        loss_fill = loss_fn_model_fill(pred_fill, target_fill)
        loss_fill.backward(retain_graph=True)
        optimizer_model_fill.step()
        optimizer_model_fill.zero_grad()
        if batch % 100 == 0:
            loss_fill, current_fill = loss_fill.item(), batch * batch_size
            # loss_detect, current_detect = loss_detect.item(), batch * batch_size
            print(
                f"loss: {loss_fill:>7f}  [{current_fill:>5d}/{size:>5d}]",
                # f"loss: {loss_detect:>7f}  [{current_detect:>5d}/{size:>5d}]",
            )


learning_rate = 1e-3
epochs = 1
optimizer_fill = torch.optim.Adam(model_fill.parameters(), lr=learning_rate)
optimizer_detect = torch.optim.Adam(model_detect.parameters(), lr=learning_rate)
loss_fn_fill = torch.nn.CrossEntropyLoss()
loss_fn_detect = torch.nn.CrossEntropyLoss()
model_fill.to(device)
model_detect.to(device)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(
        model_detect=model_detect,
        model_fill=model_fill,
        dataloader=train_dataloader,
        loss_fn_model_detect=loss_fn_detect,
        loss_fn_model_fill=loss_fn_fill,
        optimizer_model_detect=optimizer_detect,
        optimizer_model_fill=optimizer_fill,
        device=device,
    )
print("Done!")
