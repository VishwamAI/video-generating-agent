import unittest
import torch
from scripts.text_genre_encoder import TextGenreEncoder

class TestTextGenreEncoder(unittest.TestCase):
    def setUp(self):
        self.genre_list = ["action", "drama", "comedy", "horror", "sci-fi"]
        self.encoder = TextGenreEncoder(self.genre_list)
        self.sample_text = "A thrilling adventure in space."
        self.sample_genre = "sci-fi"

    def test_encode_text(self):
        text_embedding = self.encoder.encode_text(self.sample_text)
        self.assertEqual(text_embedding.shape, (1, 768))  # BERT base model output size

    def test_encode_genre(self):
        genre_embedding = self.encoder.encode_genre(self.sample_genre)
        self.assertEqual(genre_embedding.shape, (len(self.genre_list),))
        self.assertEqual(genre_embedding[self.genre_list.index(self.sample_genre)], 1.0)

    def test_encode(self):
        text_embedding, genre_embedding = self.encoder.encode(self.sample_text, self.sample_genre)
        self.assertEqual(text_embedding.shape, (1, 768))
        self.assertEqual(genre_embedding.shape, (len(self.genre_list),))
        self.assertEqual(genre_embedding[self.genre_list.index(self.sample_genre)], 1.0)

if __name__ == "__main__":
    unittest.main()
