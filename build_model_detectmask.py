import torch
import torch.nn as nn
from ultis.DataBuilderElectra import DataBuilderForElectra
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, AutoTokenizer
from ultis.CustomDataset import CustomDataset
import os
import time
import random
from torch.utils.data import TensorDataset
from model.DetectModel import DetectModel

batch_size = 150

os.makedirs("data", exist_ok=True)
## Define model and tokenizer phobert
phobert = RobertaForMaskedLM.from_pretrained("vinai/phobert-base-v2")
pho_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

## Get data
file = open("data/sentences.txt", "r")
sentences = file.readlines()
sentences = [sentence.replace("\n", "").lower() for sentence in sentences]

## Data train
data_builder = DataBuilderForElectra(
    rng=random.Random(86), seq_max_length=64, tokenizer=pho_tokenizer
)
input_original, input_masked = data_builder.build_data(sentences)
train_dataset = TensorDataset(input_original, input_masked)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DetectModel(
    embed_dim=768,
    num_heads=4,
    num_layer=6,
    dff=128,
    vocab_size=pho_tokenizer.vocab_size,
    device=device,
)


def train_loop(dataloader, model, loss_fn, optimizer, device, phobert):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (original, masked) in enumerate(dataloader):
        # Define data
        with torch.no_grad():
            attn_mask = masked != 1
            filled = torch.argmax(phobert(masked, attn_mask)[0], dim=-1)
            filled[filled == 2] = 1
        pred = model(filled)
        target = filled == original
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
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
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")
