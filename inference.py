from model.DetectModel import DetectModel
from model.SCModel import ModelSC
from ultis.Tokenizer import Tokenizer
from ultis.DataBuilder import padding
import torch


path_model_fill = ""
path_model_detect = ""

model_fill: ModelSC = torch.load(path_model_fill)
model_detect: DetectModel = torch.load(path_model_detect)


def correct_spell(text, custom_tokenizer: Tokenizer, char_tokenizer: Tokenizer):
    words_input_ids = torch.tensor(
        custom_tokenizer.convert_tokens_to_ids(custom_tokenizer.tokenize)
    )
    masked_position = words_input_ids == 2
    char_input_ids = char_tokenizer.tokenize(text=text)
    # with model_fill.eval() and model_detect.eval():
    #     model_fill
