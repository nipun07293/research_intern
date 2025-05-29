# !pip install transformers sentence-transformers rouge_score nltk
!pip install rouge_score
# Import libraries
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch

# HuggingFace Transformers for BLIP and GPT-2
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Sentence-BERT for semantic similarity
from sentence_transformers import SentenceTransformer, util

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download('punkt')  # for BLEU tokenization

# (Optional) Mount Google Drive (run in Colab if using Drive)
from google.colab import drive
drive.mount('/content/drive')
# train_csv_path = "/content/drive/MyDrive/train_set_new.csv"
# --- Setup paths ---
drive_path = "/content/drive/MyDrive"  # Base Drive path

# CSV file paths
train_csv = os.path.join(drive_path, "train_set_new.csv")
test_csv  = os.path.join(drive_path, "test_set_new.csv")

# Image directory paths
train_img_dir = os.path.join(drive_path, "training")
test_img_dir  = os.path.join(drive_path, "testing")

# Output file path
output_path = os.path.join(drive_path, "test_few_shot_results.csv")

# --- Load data ---
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# --- Add full image paths ---
train_df['image_path'] = train_df['image_name'].fillna('').astype(str).apply(lambda x: os.path.join(train_img_dir, x))
test_df['image_path'] = test_df['image_name'].fillna('').astype(str).apply(lambda x: os.path.join(test_img_dir, x))

# --- Info ---
print(f"Training examples: {len(train_df)}, Test examples: {len(test_df)}")
print("Columns in training set:", train_df.columns.tolist())
# Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
def generate_caption(image_path):
    """Generate a caption for an image using BLIP."""
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return ""
    inputs = blip_processor(images=raw_image, return_tensors="pt")
    inputs = {k: v.to(blip_model.device) for k, v in inputs.items()}
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Example: generate caption for the first training image
example_caption = generate_caption(train_df.iloc[0]['image_path'])
print("Example BLIP caption:", example_caption)
# Encode all training titles
# Encode all training titles
st_model = SentenceTransformer('all-MiniLM-L6-v2')
# Assuming the title column is named 'noisy_title' based on previous code
train_titles = train_df['noisy_title'].tolist()
train_embeddings = st_model.encode(train_titles, convert_to_tensor=True)

def get_top_k_examples(query_title, k=3):
    """Return indices of the top-k training examples semantically similar to query_title."""
    query_embedding = st_model.encode(query_title, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, train_embeddings)[0]
    top_results = torch.topk(cosine_scores, k)
    return top_results.indices.tolist()

# Example: find 3 similar titles to a sample test title
# Assuming the title column in test_df is also 'noisy_title'
sample_title = test_df.iloc[0]['noisy_title']
similar_indices = get_top_k_examples(sample_title, k=3)
# Display the similar titles using the correct column name
print("Similar training titles:", train_df.loc[similar_indices, 'noisy_title'].values)
def construct_prompt(test_title, test_caption, example_indices):
    """Build the few-shot prompt string using given example indices."""
    prompt = "Examples:\n"
    for idx in example_indices:
        ex_title = train_df.loc[idx, 'noisy_title']
        # Generate or retrieve the caption for the training image
        ex_image_path = train_df.loc[idx, 'image_path']
        ex_caption = generate_caption(ex_image_path)
        # Use the provided 2-step summary from training set
        ex_summary = train_df.loc[idx, 'summary']
        # Split the summary into two lines if needed
        summary_lines = [line.strip() for line in str(ex_summary).split('\n') if line.strip()]
        # Format summary as two bullet points
        prompt += f"Title: {ex_title}\n"
        prompt += f"Caption: {ex_caption}\n"
        prompt += "Summary:\n"
        for i, line in enumerate(summary_lines[:2], start=1):
            prompt += f"{i}. {line}\n"
        prompt += "\n"
    # Add the new recipe prompt
    prompt += f"New Recipe:\nTitle: {test_title}\n"
    prompt += f"Visual Context: {test_caption}\n"
    prompt += "Generate comprehensive 2-step cooking summary:"
    return prompt

# Example: construct prompt for the first test sample
test_title = test_df.iloc[0]['noisy_title']
test_caption = generate_caption(test_df.iloc[0]['image_path'])
topk_indices = get_top_k_examples(test_title, k=3)
prompt = construct_prompt(test_title, test_caption, topk_indices)
print(prompt[:200] + "...")  # print beginning of prompt
# Initialize the GPT-2 text generation pipeline
generator = pipeline('text-generation', model='gpt2-medium', tokenizer='gpt2-medium',
                     device=0 if torch.cuda.is_available() else -1)

def generate_summary(prompt_text, max_length=200):
    """Generate summary text from prompt using GPT-2."""
    result = generator(prompt_text, max_length=max_length, num_return_sequences=1, do_sample=False)
    generated = result[0]['generated_text']
    # Remove the prompt part to isolate the summary
    summary_text = generated.replace(prompt_text, "").strip()
    # If the model output includes a third step, cut it off
    if "3." in summary_text:
        summary_text = summary_text.split("3.")[0].strip()
    return summary_text

# Example: generate summary for the first test recipe
summary_output = generate_summary(prompt)
print("Generated Summary:\n", summary_output)
# Initialize metrics
smooth_fn = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Download the required NLTK data
import nltk
nltk.download('punkt_tab', quiet=True)

results = []
for idx in range(min(10, len(test_df))):
    test_title = test_df.loc[idx, 'noisy_title']
    test_caption = generate_caption(test_df.loc[idx, 'image_path'])
    example_indices = get_top_k_examples(test_title, k=3)
    prompt = construct_prompt(test_title, test_caption, example_indices)
    gen_summary = generate_summary(prompt)

    # Clean numbering for evaluation
    cleaned_summary = "\n".join([line.strip()[line.find('.')+1:].strip()
                                 for line in gen_summary.split('\n') if line.strip()])

    # Get reference text and ensure it's a string before processing
    reference_text_raw = test_df.loc[idx, 'orignal_instructions']
    # Convert to string, handling potential NaN values
    reference_text = str(reference_text_raw) if pd.notna(reference_text_raw) else ""


    # Compute BLEU (sentence_bleu expects tokens)
    # Ensure reference_text is not empty before tokenizing
    if reference_text:
        ref_tokens = nltk.word_tokenize(reference_text.lower())
    else:
        ref_tokens = [] # Handle empty reference string

    hyp_tokens = nltk.word_tokenize(cleaned_summary.lower())

    # Handle cases where hypothesis is empty to avoid errors in sentence_bleu
    if hyp_tokens and ref_tokens:
        bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_fn)
    else:
        bleu_score = 0.0 # Assign 0 BLEU score if either is empty

    # Compute ROUGE scores (F1)
    # Ensure both reference and candidate are non-empty strings for ROUGE
    if reference_text and cleaned_summary:
        rouge_scores = scorer.score(reference_text, cleaned_summary)
        rouge1_f = rouge_scores['rouge1'].fmeasure
        rougel_f = rouge_scores['rougeL'].fmeasure
    else:
        rouge1_f = 0.0
        rougel_f = 0.0

    results.append({
        'Title': test_title,
        'Caption': test_caption,
        'Summary': gen_summary,
        'BLEU': round(bleu_score * 100, 2),
        'ROUGE-1': round(rouge1_f * 100, 2),
        'ROUGE-L': round(rougel_f * 100, 2)
    })

    print(f"Example {idx+1} - BLEU: {bleu_score:.3f}, ROUGE-1: {rouge1_f:.3f}, ROUGE-L: {rougel_f:.3f}")
if results:
    avg_bleu = np.mean([r['BLEU'] for r in results])
    avg_rouge1 = np.mean([r['ROUGE-1'] for r in results])
    avg_rougel = np.mean([r['ROUGE-L'] for r in results])
    print(f"\nAverage BLEU: {avg_bleu:.2f}, ROUGE-1: {avg_rouge1:.2f}, ROUGE-L: {avg_rougel:.2f}")
    results_df = pd.DataFrame(results)
output_path = os.path.join(drive_path, "recipe_summaries_output1.csv")
results_df.to_csv(output_path, index=False)
print(f"Saved summaries and metrics to {output_path}")
