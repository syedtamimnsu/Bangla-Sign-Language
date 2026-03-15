# README.md
## Sign Language Recognition Pipeline
This project implements a pipeline for continuous sign language recognition using MediaPipe for landmark extraction, I3D-ResNet50 for video features, and a Transformer-based seq2seq model.

### Installation
1. Clone the repo: `git clone https://github.com/NahinAlam001/sign_language_pipeline.git`
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Install inflated_convnets_pytorch: `git clone https://github.com/hassony2/inflated_convnets_pytorch.git` and add to PYTHONPATH.

### Usage
Run `python main.py` to execute the pipeline. Configure via `config.yaml`.

### Structure
- `src/`: Core modules (extractors, dataset, model, trainer, utils).
- `config.yaml`: Parameters.
- Data: Place videos and `subset_mapping.csv` in base_dir.

### License
MIT License (see LICENSE file).

For contributions or issues, open a pull request.
