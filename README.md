# Recipe Instruction Generation and Evaluation

This project contains Python code designed to generate cooking instructions from image captions and titles using a combination of the BLIP model for image captioning and GPT-2 (medium) for text generation. It also includes code for evaluating the generated instructions using ROUGE metrics by comparing them against original reference instructions.

The code is structured to run in a Jupyter notebook environment, particularly within Google Colab, and assumes access to files stored in Google Drive.

## Project Structure

The code is broken down into several logical parts, likely corresponding to different cells in your notebook:

1.  **Setup and Imports:** Installs necessary libraries and imports modules (`transformers`, `PIL`, `pandas`, `torch`, `nltk`, `rouge_score`, `os`, `re`, `string`).
2.  **Google Drive Mounting:** Code to mount Google Drive to access data and save results.
3.  **Model Initialization:** Loads the BLIP image captioning model (`Salesforce/blip-image-captioning-base`) and a GPT-2 medium text generation model (`gpt2-medium`) using the `transformers` library, moving them to the GPU if available.
4.  **Helper Functions:**
    *   `generate_caption(path)`: Takes an image file path, generates a caption using the BLIP model.
    *   `generate_summary(caption, title)`: Takes a caption and title, generates cooking instructions using the GPT-2 model with a specific prompt and generation parameters. Includes basic formatting/re-numbering logic.
5.  **Filepaths and Data Loading:** Defines paths for image directories and CSV files in Google Drive, and loads the training and test dataframes using pandas. Includes error handling for missing files.
6.  **Data Processing (Train Subset):** Iterates through a subset of the training data, generates captions and instructions for each sample, and saves the results to a new CSV file.
7.  **Data Processing (Test Subset):** Iterates through a subset of the test data, generates captions and instructions, and saves the results to another CSV file, including the original instructions for evaluation.
8.  **Evaluation Setup:** Imports evaluation libraries (`nltk`, `rouge_score`), downloads necessary NLTK data, and defines a helper function (`clean_text_for_rouge`) for text preprocessing before evaluation.
9.  **Evaluation (ROUGE):** Loads the generated test results CSV, prepares the reference (original) and candidate (generated) texts, calculates average ROUGE-1, ROUGE-2, and ROUGE-L F-measures, Precision, and Recall using the `rouge_score` library, and prints the results.

## Requirements

*   Python 3.7+
*   Jupyter Notebook or Google Colab environment
*   Access to Google Drive (for data storage and saving results)
*   Libraries: `transformers`, `Pillow` (PIL), `pandas`, `torch`, `nltk`, `rouge_score`

## Setup

1.  **Install Libraries:** Run the following commands in your notebook cells (typically near the beginning):
   ## Usage

Execute the code cells in your Jupyter notebook sequentially.

1.  Run the installation and import cells.
2.  Run the Google Drive mounting cell.
3.  Run the cell initializing the models and defining the helper functions.
4.  Run the cell defining file paths, loading data, and processing the training/test subsets. This will generate `train_improved_summaries.csv` and `test_improved_summaries.csv` in your Google Drive.
5.  Run the cell setting up the evaluation (imports, NLTK downloads, `clean_text_for_rouge`).
6.  Run the cell that loads the test results CSV (`test_improved_summaries.csv`) and calculates/prints the ROUGE scores.

