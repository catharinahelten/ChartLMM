

# Image Question-Answering with Pix2Struct and MatCha 

## Project Description
This project shows the usage of two Google Large Multimodal Models (Pix2Struct and MatCha) from the Hugging Face Transformers library to perform question-answering on images. 
The project loads chart images from a specified directory and uses two different pre-trained models to generate responses to questions provided in a text file.
The purpose of the project is to compare the performance of Pix2Struct and MatCha. MatCha is build on top of Pix2Struct Base model, finetuned with chart-images as training data.

## Features
- Processes images and generates responses to questions using Pix2Struct and MatCha models.
- Supports multiple image formats (PNG, JPG, JPEG).
- Includes an evaluation folder with an answer ground truth text file for comparing the outputs with their ground truth values.

## Installation
### Prerequisites
- Python (>=3.6)
- Hugging Face Transformers
- PIL (Python Imaging Library)

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/catharinahelten/ChartLMM.git
    ```

2. Navigate to the project directory:
    ```bash
    cd ChartLMM
    ```

3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Usage
1. Ensure that your image files are placed in the `images` directory.
2. Provide your questions in the `questions.txt` file, with each question on a new line.
3. Open terminal, go to project folder and type: python3 LMM.py
4. Use the `evaluation/ground_truth.txt` file to compare the generated responses with their ground truth values.

