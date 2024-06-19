import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class TextGenreEncoder:
    def __init__(self, genre_list):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.genre_list = genre_list
        self.genre_dict = {genre: idx for idx, genre in enumerate(genre_list)}

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        text_embedding = outputs.last_hidden_state.mean(dim=1)
        return text_embedding

    def encode_genre(self, genre):
        genre_vector = np.zeros(len(self.genre_list))
        if genre in self.genre_dict:
            genre_vector[self.genre_dict[genre]] = 1
        return torch.tensor(genre_vector, dtype=torch.float32)

    def encode(self, text, genre):
        text_embedding = self.encode_text(text)
        genre_embedding = self.encode_genre(genre)
        return text_embedding, genre_embedding

if __name__ == "__main__":
    genre_list = ["action", "drama", "comedy", "horror", "sci-fi"]
    encoder = TextGenreEncoder(genre_list)

    sample_text = "A thrilling adventure in space."
    sample_genre = "sci-fi"

    text_embedding, genre_embedding = encoder.encode(sample_text, sample_genre)
    print("Text Embedding:", text_embedding)
    print("Genre Embedding:", genre_embedding)
