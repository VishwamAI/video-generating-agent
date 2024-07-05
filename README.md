# Video-generating-agent

## Project Overview

The `video-generating-agent` project aims to generate videos using advanced neural network models. It leverages state-of-the-art techniques to create photorealistic videos from various input data sources.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (optional but recommended)
- [FFMPEG](https://ffmpeg.org/download.html) (for video processing)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/VishwamAI/video-generating-agent.git
   cd video-generating-agent
   ```

2. (Optional) Install Miniconda:
   Follow the instructions on the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) to install Miniconda.

3. Set up the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate video-generating-agent
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure Kaggle API credentials:
   Place your `kaggle.json` file in the `~/.kaggle` directory. Ensure it contains the following:
   ```json
   {
     "username": "kasinadhsarma1",
     "key": "5df3fbdf09edab914b50663965c9f4f2"
   }
   ```

## Usage

### Preprocessing

1. Preprocess the dataset:
   ```bash
   python preprocess.py --input data/example_sequence/
   ```

### Training

1. Train the model with the scene-specific config:
   ```bash
   python train.py --config configs/training_config.txt
   ```

### Free Viewpoint Rendering

1. Synthesize a novel camera path:
   ```bash
   python free_viewpoint_rendering.py --input experiments/experiment_1/ --deformations train --camera_path fixed --fixed_view 10
   ```

## CI/CD Pipeline

The CI/CD pipeline is configured to automate the following steps:

1. Install dependencies
2. Download and verify datasets
3. Preprocess datasets
4. Train the model
5. Verify the presence of image files
6. Run tests and checks

The pipeline configuration file is located at `.github/workflows/ci.yml`.

## Contributing

We welcome contributions to the project. Please follow these guidelines:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
