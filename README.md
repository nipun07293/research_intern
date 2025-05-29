Recipe Summarization Pipeline

Overview

This project implements a multimodal few-shot learning pipeline to generate concise, 2-step cooking instructions from a recipe's image and title. The system combines state-of-the-art models for image captioning, semantic retrieval, and language generation.

Pipeline Components

1. Image Captioning (BLIP)

Model: Salesforce's BLIP (Bootstrapped Language Image Pretraining)

Purpose: Generate natural-language descriptions of the recipe image to provide visual context.

2. Semantic Retrieval (Sentence-BERT)

Model: all-MiniLM-L6-v2

Purpose: Retrieve top-3 semantically similar training examples based on the recipe title using cosine similarity over embeddings.

3. Prompt Construction

Each prompt includes:

3 examples, each with:

Title

BLIP-generated caption

2-step summary

Target test case with:

Title

Caption

Instruction: "Generate comprehensive 2-step cooking summary"

4. Text Generation (GPT-2 Medium)

Model: HuggingFace's gpt2-medium

Usage: Generate 2-step instructions using the prompt in a few-shot manner (no model fine-tuning).

Setup Instructions


2. Prepare Data

Place training and testing CSV files in the specified directories.

Ensure image paths are correct and accessible.

3. Run Pipeline

Execute the script to:

Generate image captions

Retrieve relevant examples

Construct few-shot prompts

Generate and evaluate summaries

Save outputs to a CSV file

Evaluation

The output summaries are evaluated using:

BLEU (token overlap)

ROUGE-1 (unigram recall)

ROUGE-L (longest common subsequence)



Known Limitations

Occasional hallucination or omission of recipe steps.

Formatting inconsistency in GPT-2 output.

BLEU is low due to GPT-2’s abstractiveness.

Contributions

Prompt engineering and structure design.

Integration of vision-language and retrieval models.

Evaluation and result analysis.
