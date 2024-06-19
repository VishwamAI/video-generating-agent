import unittest
import torch
from scripts.text_genre_encoder import TextGenreEncoder
from nonrigid_nerf.train import training_wrapper_class, create_nerf

class TestTrainingWrapperClass(unittest.TestCase):
    def setUp(self):
        # Initialize the TextGenreEncoder with a genre list
        self.genre_list = ["Action", "Drama", "Comedy", "Horror", "Sci-Fi"]
        self.text_genre_encoder = TextGenreEncoder(self.genre_list)

        # Create dummy models and latents
        self.coarse_model = torch.nn.Linear(10, 10)
        self.latents = [torch.randn(10) for _ in range(5)]
        self.fine_model = torch.nn.Linear(10, 10)
        self.ray_bender = torch.nn.Linear(10, 10)

        # Initialize the training wrapper class
        self.wrapper = training_wrapper_class(
            self.coarse_model, self.latents, self.genre_list, self.fine_model, self.ray_bender
        )

    def test_forward(self):
        # Create dummy inputs
        args = type('', (), {})()
        args.chunk = 1024
        args.offsets_loss_weight = 0.0
        args.divergence_loss_weight = 0.0
        args.N_samples = 64
        args.N_importance = 0
        args.N_iters = 1000

        rays_o = torch.randn(1024, 3)
        rays_d = torch.randn(1024, 3)
        i = 0
        render_kwargs_train = {}
        target_s = torch.randn(1024, 3)
        global_step = 0
        start = 0
        dataset_extras = {"imageid_to_timestepid": [0, 1, 2, 3, 4]}
        batch_pixel_indices = torch.randint(0, 5, (1024, 2))
        text = "Sample text description"
        genre = "Action"

        # Call the forward method
        loss = self.wrapper.forward(
            args, rays_o, rays_d, i, render_kwargs_train, target_s, global_step, start, dataset_extras, batch_pixel_indices, text, genre
        )

        # Check that the loss is a tensor
        self.assertIsInstance(loss, torch.Tensor)

    def test_text_genre_encoding(self):
        # Test text and genre encoding
        text = "This is a test sentence."
        genre = "Action"
        text_embedding, genre_embedding = self.text_genre_encoder.encode(text, genre)
        self.assertIsNotNone(text_embedding)
        self.assertIsNotNone(genre_embedding)
        self.assertEqual(text_embedding.shape[0], 1)  # Batch size should be 1
        self.assertEqual(genre_embedding.shape[0], len(self.genre_list))  # Genre vector length should match genre list length

    def test_integration_with_additional_pixel_information(self):
        # Create dummy inputs
        args = type('', (), {})()
        args.chunk = 1024
        args.offsets_loss_weight = 0.0
        args.divergence_loss_weight = 0.0
        args.N_samples = 64
        args.N_importance = 0
        args.N_iters = 1000

        rays_o = torch.randn(1024, 3)
        rays_d = torch.randn(1024, 3)
        i = 0
        render_kwargs_train = {}
        target_s = torch.randn(1024, 3)
        global_step = 0
        start = 0
        dataset_extras = {"imageid_to_timestepid": [0, 1, 2, 3, 4]}
        batch_pixel_indices = torch.randint(0, 5, (1024, 2))
        text = "Sample text description"
        genre = "Action"

        # Encode text and genre
        text_embedding, genre_embedding = self.text_genre_encoder.encode(text, genre)

        # Call the forward method
        loss = self.wrapper.forward(
            args, rays_o, rays_d, i, render_kwargs_train, target_s, global_step, start, dataset_extras, batch_pixel_indices, text, genre
        )

        # Check that the loss is a tensor
        self.assertIsInstance(loss, torch.Tensor)
        # Check that the text_genre_latents are included in the additional_pixel_information
        self.assertIn("text_genre_latents", self.wrapper.additional_pixel_information)
        self.assertTrue(torch.equal(self.wrapper.additional_pixel_information["text_genre_latents"], text_embedding))

if __name__ == "__main__":
    unittest.main()
