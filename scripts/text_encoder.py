import torch
from transformers import BertTokenizer, BertModel

class TextEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, text, genre):
        combined_text = f"{genre}: {text}"
        inputs = self.tokenizer(combined_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode text into latent representation using BERT.")
    parser.add_argument("text", type=str, help="Text description to encode.")
    parser.add_argument("genre", type=str, help="Genre of the text description.")
    args = parser.parse_args()

    encoder = TextEncoder()
    latent_representation = encoder.encode(args.text, args.genre)
    print(latent_representation)
