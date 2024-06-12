import torch
from DataBuilder import DataBuilderForElectra
from transformers import AutoTokenizer, RobertaForMaskedLM
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from DetectModel import DetectModel
from CustomDataset import CustomDataset
import random

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = RobertaForMaskedLM.from_pretrained("vinai/phobert-base-v2")
device = torch.device("cpu")
modelElectraSC = DetectModel(
    embed_dim=768,
    num_heads=4,
    dff=128,
    phobert_embedding=phobert.embeddings,
    num_layer=6,
    device=device,
)
rng = random.Random(123)
PATH_FILE_DATA = ""
file = open(PATH_FILE_DATA, "r")
sentences = file.readlines()
data_builder = DataBuilderForElectra(
    tokenizer.mask_token,
    tokenizer.pad_token,
    tokenizer.sep_token,
    tokenizer.cls_token,
    tokenizer.eos_token,
    64,
    rng,
)
input_masked, token_masked_positions, token_masked_labels = (
    data_builder.build_data_for_generator(sentences)
)
target = data_builder.build_target(sentences)

input_masked_ids = torch.tensor(
    [tokenizer.convert_tokens_to_ids(s) for s in input_masked]
)
target_ids = torch.tensor([tokenizer.convert_tokens_to_ids(s) for s in target])
phobert.eval()
batch_size = 100
epoch = 5
loss_fn = torch.nn.CrossEntropyLoss()
data_train = {
    "input_ids": input_masked_ids,
    "target": target_ids,
}
dataset = CustomDataset(data_train)
train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


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
