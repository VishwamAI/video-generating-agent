import torch
from scripts.text_genre_encoder import TextGenreEncoder
from nonrigid_nerf.train import training_wrapper_class, render_wrapper_class, render, render_rays, create_nerf

# Initialize the TextGenreEncoder with a list of genres
genre_list = ["action", "drama", "comedy", "horror", "sci-fi"]
encoder = TextGenreEncoder(genre_list)

# Sample text and genre for testing
sample_text = "A thrilling adventure in space."
sample_genre = "sci-fi"

# Encode the sample text and genre
text_embedding, genre_embedding = encoder.encode(sample_text, sample_genre)

# Print the embeddings for verification
print("Text Embedding:", text_embedding)
print("Genre Embedding:", genre_embedding)

# Modify the training_wrapper_class to include text and genre encoding
class ModifiedTrainingWrapperClass(training_wrapper_class):
    def forward(
        self,
        args,
        rays_o,
        rays_d,
        i,
        render_kwargs_train,
        target_s,
        global_step,
        start,
        dataset_extras,
        batch_pixel_indices,
    ):
        # Generate text and genre embeddings
        text_embedding, genre_embedding = encoder.encode(sample_text, sample_genre)

        # Include the embeddings in additional_pixel_information
        additional_pixel_information = {
            "text_embedding": text_embedding,
            "genre_embedding": genre_embedding,
        }

        # Call the original forward method with the modified additional_pixel_information
        return super().forward(
            args,
            rays_o,
            rays_d,
            i,
            render_kwargs_train,
            target_s,
            global_step,
            start,
            dataset_extras,
            batch_pixel_indices,
            additional_pixel_information=additional_pixel_information,
        )

# Instantiate the modified training wrapper class
modified_training_wrapper = ModifiedTrainingWrapperClass(
    coarse_model=None,  # Replace with actual model
    latents=None,       # Replace with actual latents
    fine_model=None,    # Replace with actual fine model if available
    ray_bender=None     # Replace with actual ray bender if available
)

# Example usage of the modified training wrapper class
# Replace the following variables with actual values
args = None
rays_o = None
rays_d = None
i = None
render_kwargs_train = None
target_s = None
global_step = None
start = None
dataset_extras = None
batch_pixel_indices = None

# Call the forward method of the modified training wrapper class
output = modified_training_wrapper.forward(
    args,
    rays_o,
    rays_d,
    i,
    render_kwargs_train,
    target_s,
    global_step,
    start,
    dataset_extras,
    batch_pixel_indices,
)

# Print the output for verification
print("Output:", output)
