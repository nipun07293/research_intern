from google.colab import drive
# Modify the mount call to potentially force a remount if the default fails
# If the initial mount fails, try uncommenting the line below:
# drive.mount('/content/drive', force_remount=True)
# Otherwise, keep the standard mount call:
drive.mount('/content/drive')

import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
import re # Import the regular expression module

# 1) Model setup
# Check if GPU is available and use it
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Using base models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device) # Move model to device

# Using gpt2-medium again, as it might follow instructions better than base gpt2
# (Note: this increases resource usage compared to base gpt2)
text_generator = pipeline("text-generation", model="gpt2-medium", device=0 if torch.cuda.is_available() else -1) # Use gpt2-medium

# 2) Helpers
def generate_caption(path):
    try:
        img = Image.open(path).convert("RGB")
        # Move inputs to the same device as the model
        inputs = processor(img, return_tensors="pt").to(device)
        # Generate caption, slightly longer max_length for more detail
        out = blip_model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        return processor.decode(out[0], skip_special_tokens=True)
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return "Error: Image not found"
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return f"Error: {e}"


def generate_summary(caption, title):
    # Refined prompt asking for detailed, comprehensive instructions
    prompt = (
        f"Title: {title}\n"
        f"Image Description: {caption}\n"
        f"Write comprehensive step-by-step cooking instructions for this recipe, including ingredients and preparation details. Format as numbered steps:\n"
        f"1." # Encourage the model to start with step 1
    )

    # Adjusted generation parameters for more detailed output
    gen = text_generator(
        prompt,
        max_length=250, # Increased max length significantly
        num_return_sequences=1,
        do_sample=False, # Keep non-sampling
        num_beams=4,     # Use beam search
        early_stopping=True, # Stop when all beam hypotheses have a stop token
        pad_token_id=text_generator.model.config.eos_token_id,
        eos_token_id=text_generator.model.config.eos_token_id,
    )[0]["generated_text"]

    # Extract the generated part after the prompt
    split_key = f"Write comprehensive step-by-step cooking instructions for this recipe, including ingredients and preparation details. Format as numbered steps:\n"
    summary_part = gen.split(split_key)[-1].strip()

    # Attempt a basic cleanup/re-numbering for better structure, but keep it relatively simple
    lines = summary_part.split('\n')
    cleaned_lines = []
    step_counter = 1
    # Regex to match a line starting with a number, optionally followed by . or ), then optional whitespace
    # This regex has only one capturing group for the number (\d+)
    step_start_pattern = re.compile(r'^\s*\d+[.\)]?\s*') # Added optional leading whitespace

    for line in lines:
        line = line.strip()
        if not line: # Skip empty lines
            continue

        # Use search to find the step number pattern at the beginning of the line
        match = step_start_pattern.search(line)

        if match:
            # If a step pattern is found, extract the text *after* the match
            text_after_step = line[match.end():].strip()
            # Append with the correct step counter
            cleaned_lines.append(f"{step_counter}. {text_after_step}")
            step_counter += 1
        else:
             # If the line doesn't start with a recognized step number pattern,
             # just append it as a continuation or potentially a new unnumbered step.
             # For simplicity and to avoid losing text, we'll just append it with the next number.
             # A more sophisticated parser might try to merge these into the previous step.
             cleaned_lines.append(f"{step_counter}. {line}")
             step_counter += 1 # Increment anyway to avoid huge single steps


    # Join the cleaned lines. Limit the number of steps to keep output manageable.
    # The original code limited to 5 steps, let's keep that for consistency with the provided code state.
    formatted_summary = "\n".join(cleaned_lines[:5])

    # Fallback: If cleaning produces nothing, return the raw split text
    if not formatted_summary and summary_part:
        return summary_part
    elif not formatted_summary: # If even the raw split text is empty
        return "Summary generation failed."

    return formatted_summary


# 3) Filepaths & CSVs
# Update image directory paths to point to Google Drive
# Assuming 'training' and 'testing' folders are directly within your Google Drive's "My Drive"
TRAIN_IMG_DIR = "/content/drive/MyDrive/training"
TEST_IMG_DIR  = "/content/drive/MyDrive/testing"
TRAIN_CSV     = "/content/drive/MyDrive/train_set1.csv" # Ensure this points to your train CSV
TEST_CSV      = "/content/drive/MyDrive/test_set2.csv"   # Ensure this points to your test CSV

# Read both dataframes
try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded train set with {len(train_df)} rows")
    print(f"Loaded test set with {len(test_df)} rows")
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    print("Please ensure train_set1.csv and test_set2.csv are in your Google Drive's 'My Drive' folder.")
    train_df = pd.DataFrame() # Create empty dataframes to avoid errors
    test_df = pd.DataFrame()


# 4) Process TRAIN set (Subset) → train_improved_summaries.csv
train_out = []
if not train_df.empty:
    print("\nProcessing training set subset...")
    # Process only the first 5 rows for quick testing
    train_subset_df = train_df.head(10)
    print(f"Processing a subset of {len(train_subset_df)} train samples.")

    for index, row in train_subset_df.iterrows(): # Iterate over the train subset
        img_path = os.path.join(TRAIN_IMG_DIR, row["image_name"])
        print(f"Processing {img_path} ({index+1}/{len(train_subset_df)})...")
        caption = generate_caption(img_path)

        summary = "" # Initialize summary
        # Only attempt to generate summary if caption generation was successful
        if not caption.startswith("Error:"):
             # Generate summary using the improved prompt/params
             summary = generate_summary(caption, row["noisy_title"])
        else:
            summary = "Summary not generated due to caption error."


        train_out.append({
            "image_name":        row["image_name"],
            "noisy_title":       row["noisy_title"],
            "full_instructions": row.get("full_instructions", "N/A"), # Get instructions if column exists, default to N/A
            "caption":           caption,
            "generated_summary": summary # Added generated summary to train output
        })

    # Define BASE path for saving output files
    BASE = "/content/drive/MyDrive/"

    train_output_path = os.path.join(BASE, "train_improved_summaries.csv") # New output filename
    pd.DataFrame(train_out).to_csv(
        train_output_path,
        index=False
    )
    print(f"→ Wrote training set results to {train_output_path}")
else:
    print("Training dataframe is empty. Skipping training set processing.")


# 5) Process TEST set (Subset) → test_improved_summaries.csv
test_out = []
if not test_df.empty:
    print("\nProcessing test set subset...")
    # Process only the first 5 rows for quick testing
    test_subset_df = test_df.head(7)
    print(f"Processing a subset of {len(test_subset_df)} test samples.")

    for index, row in test_subset_df.iterrows(): # Iterate over the test subset
        img_path = os.path.join(TEST_IMG_DIR, row["image_name"])
        print(f"Processing {img_path} ({index+1}/{len(test_subset_df)})...")
        caption = generate_caption(img_path)

        summary = "" # Initialize summary
        # Only attempt to generate summary if caption generation was successful
        if not caption.startswith("Error:"):
            # Generate summary using the improved prompt/params
            summary = generate_summary(caption, row["noisy_title"])
        else:
            summary = "Summary not generated due to caption error."


        test_out.append({
            "image_name":        row["image_name"],
            "noisy_title":       row["noisy_title"],
            "original_instructions": row.get("full_instructions", "N/A"), # Get instructions if column exists, default to N/A
            "caption":           caption,
            "generated_summary": summary
        })

    # Define BASE path for saving output files
    BASE = "/content/drive/MyDrive/"

    # Update output filename
    test_output_path = os.path.join(BASE, "test_improved_summaries1.csv") # New output filename
    pd.DataFrame(test_out).to_csv(
        test_output_path,
        index=False
    )
    print(f"→ Wrote test set results to {test_output_path}")
else:
     print("Test dataframe is empty. Skipping test set processing.")
  import nltk
# Removed corpus_bleu
from rouge_score import rouge_scorer
import pandas as pd
import os
import string # Import string for punctuation removal

# Download necessary NLTK data (if not already downloaded)
# Keep existing downloads. 'punkt' is essential for tokenization.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found, downloading...")
    nltk.download('punkt')
except Exception as e:
    print(f"An unexpected error occurred while checking/downloading NLTK 'punkt': {e}")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
     print("NLTK 'wordnet' not found, downloading...")
     nltk.download('wordnet')
except Exception as e:
    print(f"An unexpected error occurred while checking/downloading NLTK 'wordnet': {e}")

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("NLTK 'omw-1.4' not found, downloading...")
    nltk.download('omw-1.4')
except Exception as e:
     print(f"An unexpected error occurred while checking/downloading NLTK 'omw-1.4': {e}")

# Helper function for basic text cleaning (still needed for ROUGE stemmer)
# No longer needs to return tokens as ROUGE scorer takes strings
def clean_text_for_rouge(text):
    """
    Basic cleaning: remove punctuation and convert to lower case.
    RougeScorer with use_stemmer=True handles stemming internally.
    """
    if pd.isna(text) or not isinstance(text, str):
        return "" # Return empty string for invalid input

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (this is a simple way, may need refinement)
    # Added handling for dash within words if necessary, but simple removal is often okay for ROUGE
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()


# --- Ensure this part is run AFTER the test data processing cell ---
# This is the BASE path where your CSV files are saved
BASE_PATH = "/content/drive/MyDrive/"
# --- Make sure the filename matches the one saved in the previous cell ---
test_results_filename = "test_improved_summaries.csv"
test_results_path = os.path.join(BASE_PATH, test_results_filename)

print(f"Attempting to load test results from: {test_results_path}")

test_results_df = pd.DataFrame() # Initialize empty DataFrame

# Check if the test results file exists before trying to read it
if not os.path.exists(test_results_path):
    print(f"Error: Test results file not found at {test_results_path}")
    print("Please ensure the test processing cell was run successfully and the file name is correct.")
else:
    try:
        # Attempt to read the CSV, trying different encodings if necessary
        try:
            test_results_df = pd.read_csv(test_results_path)
        except UnicodeDecodeError:
            print("UnicodeDecodeError encountered, trying 'latin-1' encoding...")
            try:
                test_results_df = pd.read_csv(test_results_path, encoding='latin-1')
            except Exception as e:
                print(f"Failed to read CSV with 'latin-1' encoding: {e}")
                print("Please check the file encoding and content for issues.")

        if not test_results_df.empty:
             print(f"Successfully loaded {len(test_results_df)} rows from {test_results_path}")
        else:
             print(f"Loaded CSV file is empty or failed to load correctly.")

    except FileNotFoundError: # This should be caught by os.path.exists, but kept as a safeguard
        print(f"Error: File not found during read_csv: {test_results_path}")
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")


# references = [] # Removed BLEU lists
# candidates = [] # Removed BLEU lists

# Check if the necessary columns exist before proceeding with evaluation
# You MUST have 'original_instructions' in your test results CSV to evaluate ROUGE
required_cols_for_eval = ["original_instructions", "generated_summary"]

if not test_results_df.empty and all(col in test_results_df.columns for col in required_cols_for_eval):
    print("\nPreparing data for ROUGE evaluation...")

    # Prepare lists of strings for ROUGE evaluation
    # Filter out rows where original or generated summary is empty or an error message
    rouge_data = [(clean_text_for_rouge(row.get("original_instructions", "")),
                   clean_text_for_rouge(row.get("generated_summary", "")))
                  for index, row in test_results_df.iterrows()
                  if pd.notna(row.get("original_instructions")) and
                     pd.notna(row.get("generated_summary")) and
                     not str(row.get("generated_summary", "")).startswith("Summary not generated")]

    # Filter out pairs where cleaning resulted in empty strings
    rouge_data = [(ref, cand) for ref, cand in rouge_data if ref and cand]


    rouge_references_str = [item[0] for item in rouge_data]
    rouge_candidates_str = [item[1] for item in rouge_data]

    print(f"Evaluating {len(rouge_data)} valid samples for ROUGE.")

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_rouge_scores = []

    if rouge_references_str and rouge_candidates_str and len(rouge_references_str) == len(rouge_candidates_str):
        print("\nCalculating ROUGE scores...")
        for ref_str, cand_str in zip(rouge_references_str, rouge_candidates_str):
             # Ensure both reference and candidate are not empty strings for scoring (should be covered by list comprehension)
             # Add a check for empty strings *after* cleaning, just in case.
             if ref_str and cand_str:
                 scores = scorer.score(ref_str, cand_str)
                 all_rouge_scores.append(scores)


        # Calculate average ROUGE scores
        if all_rouge_scores:
            avg_rouge1_f = sum(score['rouge1'].fmeasure for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rouge2_f = sum(score['rouge2'].fmeasure for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rougeL_f = sum(score['rougeL'].fmeasure for score in all_rouge_scores) / len(all_rouge_scores)

            # Optionally also print precision and recall
            avg_rouge1_p = sum(score['rouge1'].precision for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rouge1_r = sum(score['rouge1'].recall for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rouge2_p = sum(score['rouge2'].precision for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rouge2_r = sum(score['rouge2'].recall for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rougeL_p = sum(score['rougeL'].precision for score in all_rouge_scores) / len(all_rouge_scores)
            avg_rougeL_r = sum(score['rougeL'].recall for score in all_rouge_scores) / len(all_rouge_scores)


            print(f"Average ROUGE-1 F-measure: {avg_rouge1_f:.4f} (P: {avg_rouge1_p:.4f}, R: {avg_rouge1_r:.4f})")
            print(f"Average ROUGE-2 F-measure: {avg_rouge2_f:.4f} (P: {avg_rouge2_p:.4f}, R: {avg_rouge2_r:.4f})")
            print(f"Average ROUGE-L F-measure: {avg_rougeL_f:.4f} (P: {avg_rougeL_p:.4f}, R: {avg_rougeL_r:.4f})")
        else:
            print("No valid ROUGE scores calculated (all samples skipped).")
    else:
        print("Insufficient valid data to calculate ROUGE scores.")

else:
    if test_results_df.empty:
         print("\nSkipping evaluation as test results dataframe is empty.")
    elif not all(col in test_results_df.columns for col in required_cols_for_eval):
         print(f"\nSkipping evaluation: Test results dataframe is missing required columns for evaluation.")
         print(f"Needed: {required_cols_for_eval}, Found: {test_results_df.columns.tolist()}")
         print("Please ensure 'original_instructions' column is included when saving the test results CSV.")
