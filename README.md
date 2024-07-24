# Spelling Corrector Based on ElectraBERT

## Introduction

This project implements a spelling corrector using the ElectraBERT model. ElectraBERT is a state-of-the-art transformer-based model known for its efficiency and effectiveness in natural language processing tasks. The spelling corrector leverages the architecture of the ElectraBERT model to identify and correct spelling errors in input text.

Use the Discriminator of ElectraBERT's architecture to detect spelling errors and use the Generator to correct these errors.

## Features

- **Accurate Spelling Correction**: Utilizes the powerful ElectraBERT model to provide highly accurate corrections for individual and language-specific misspelled words.
- **Context-Aware**: Corrects spelling mistakes based on the context of the surrounding text, ensuring meaningful and relevant suggestions.

## Requirements

- Python 3.10+
- PyTorch
- Other dependencies listed in `requirements.txt`

<!-- ## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spelling-corrector-electrabert.git
   cd spelling-corrector-electrabert
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

1. Load the pre-trained model and tokenizer:

   ```python
   from transformers import ElectraTokenizer, ElectraForMaskedLM
   import torch

   tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
   model = ElectraForMaskedLM.from_pretrained('google/electra-base-discriminator')
   ```

2. Define the correction function:

   ```python
   def correct_spelling(text):
       inputs = tokenizer(text, return_tensors="pt")
       with torch.no_grad():
           logits = model(**inputs).logits
       predicted_token_ids = torch.argmax(logits, dim=-1)
       predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0])
       return tokenizer.convert_tokens_to_string(predicted_tokens)

   text = "This is a smaple text with speling erors."
   corrected_text = correct_spelling(text)
   print(corrected_text)
   ```

### Integration

To integrate the spelling corrector into your application, simply import the `correct_spelling` function and use it to process your text data.

## Evaluation

To evaluate the performance of the spelling corrector, use the provided evaluation script. The script compares the corrected text with a ground truth dataset and calculates metrics such as accuracy and F1 score.

1. Prepare your evaluation dataset in a CSV file with columns `original_text` and `corrected_text`.

2. Run the evaluation script:
   ```bash
   python evaluate.py --dataset path/to/your/dataset.csv
   ```

## Contributing

We welcome contributions to the project. If you find a bug or have a feature request, please open an issue. For code contributions, fork the repository and submit a pull request with your changes.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [ElectraBERT](https://github.com/google-research/electra) -->
