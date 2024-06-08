import json
import os
import pandas as pd

special_char = {
    "x": "xs",
    "s": "sx",
    "d": "drg",
    "r": "rdg",
    "g": "grd",
    "ch": "tr",
    "tr": "ch",
    "n": "l",
    "l": "n",
}

vietnamese_characters = [
    "á",
    "à",
    "ả",
    "ã",
    "ạ",
    "ă",
    "ắ",
    "ằ",
    "ẳ",
    "ẵ",
    "ặ",
    "â",
    "ấ",
    "ầ",
    "ẩ",
    "ẫ",
    "ậ",
    "é",
    "è",
    "ẻ",
    "ẽ",
    "ẹ",
    "ê",
    "ế",
    "ề",
    "ể",
    "ễ",
    "ệ",
    "í",
    "ì",
    "ỉ",
    "ĩ",
    "ị",
    "ó",
    "ò",
    "ỏ",
    "õ",
    "ọ",
    "ô",
    "ố",
    "ồ",
    "ổ",
    "ỗ",
    "ộ",
    "ơ",
    "ớ",
    "ờ",
    "ở",
    "ỡ",
    "ợ",
    "ú",
    "ù",
    "ủ",
    "ũ",
    "ụ",
    "ư",
    "ứ",
    "ừ",
    "ử",
    "ữ",
    "ự",
    "ý",
    "ỳ",
    "ỷ",
    "ỹ",
    "ỵ",
    "đ",
]
# sac, huyen, hoi, nga, nang, mu, rau, mu nguoc :>
special_token_sign_vietnamese = {
    "sac": "<´>",
    "huyen": "<`>",
    "hoi": "<?>",
    "nga": "<~>",
    "nang": "<.>",
    "mu": "<^>",
    "rau": "<,>",
    "mu_nguoc": "<^^>",
}
convert_char_vietnamese = {
    "á": ("a"),
    "à": ("a"),
    "ả": ("a"),
    "ã": ("a"),
    "ạ": ("a"),
    "ă": ("a"),
    "ắ": ("a"),
    "ằ": ("a"),
    "ẳ": ("a"),
    "ẵ": ("a"),
    "ặ": ("a"),
    "â": ("a"),
    "ấ": ("a"),
    "ầ": ("a"),
    "ẩ": ("a"),
    "ẫ": ("a"),
    "ậ": ("a"),
    "é": ("e"),
    "è": ("e"),
    "ẻ": ("e"),
    "ẽ": ("e"),
    "ẹ": ("e"),
    "ê": ("e"),
    "ế": ("e"),
    "ề": ("e"),
    "ể": ("e"),
    "ễ": ("e"),
    "ệ": ("e"),
    "í": ("i"),
    "ì": ("i"),
    "ỉ": ("i"),
    "ĩ": ("i"),
    "ị": ("i"),
    "ó": ("o"),
    "ò": ("o"),
    "ỏ": ("o"),
    "õ": ("o"),
    "ọ": ("o"),
    "ô": ("o"),
    "ố": ("o"),
    "ồ": ("o"),
    "ổ": ("o"),
    "ỗ": ("o"),
    "ộ": ("o"),
    "ơ": ("o"),
    "ớ": ("o"),
    "ờ": ("o"),
    "ở": ("o"),
    "ỡ": ("o"),
    "ợ": ("o"),
    "ú": ("u"),
    "ù": ("u"),
    "ủ": ("u"),
    "ũ": ("u"),
    "ụ": ("u"),
    "ư": ("u"),
    "ứ": ("u"),
    "ừ": ("u"),
    "ử": ("u"),
    "ữ": ("u"),
    "ự": ("u"),
    "ý": ("y"),
    "ỳ": ("y"),
    "ỷ": ("y"),
    "ỹ": ("y"),
    "ỵ": ("y"),
    "đ": ("d"),
}


def write_dict_data_to_file(file_path, data, indent=2):
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=indent)


def write_list_data_to_file(file_path, list_data):
    # os.makedirs(file_path, exist_ok=True)
    with open(file=file_path, mode="w", encoding="utf-8") as file:
        file.write("".join([x + "\n" for x in list_data]))