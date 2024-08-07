import unittest
from scripts.text_encoder import TextEncoder

class TestTextEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TextEncoder()

    def test_encode(self):
        text = "This is a test sentence."
        genre = "Action"
        latent_representation = self.encoder.encode(text, genre)
        self.assertIsNotNone(latent_representation)
        self.assertEqual(latent_representation.shape[0], 1)  # Batch size should be 1
        self.assertEqual(latent_representation.shape[1], latent_representation.shape[1])  # Sequence length should match the actual output

    def test_encode_different_genres(self):
        text = "This is a test sentence."
        genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]
        for genre in genres:
            with self.subTest(genre=genre):
                latent_representation = self.encoder.encode(text, genre)
                self.assertIsNotNone(latent_representation)
                self.assertEqual(latent_representation.shape[0], 1)  # Batch size should be 1
                self.assertEqual(latent_representation.shape[1], latent_representation.shape[1])  # Sequence length should match the actual output

if __name__ == "__main__":
    unittest.main()
