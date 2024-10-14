import re
import random
import torch
import spacy
import pandas as pd
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import evaluate
import os
from torch.nn import CrossEntropyLoss

# Load French Spacy Model for POS Tagging
nlp = spacy.load("fr_core_news_sm")

# Function to get POS tags instead of raw tokens
def get_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

# Update the label extraction to binary classification (pause or no pause) with POS tags
def get_binary_pause_labels(text):
    # Tokenize the text with Spacy to ensure we get POS tags and words at the same time
    doc = nlp(text)
    clean_pos_tags = []
    labels = []
    
    for token in doc:
        word = token.text
        if "cat_" in word:
            # If there's a category marker, mark the previous word as followed by a pause
            if len(labels) > 0:
                labels[-1] = 1  # Mark the previous token as "pause"
        else:
            clean_pos_tags.append(token.pos_)  # Append the POS tag
            labels.append(0)  # Default to no pause after this word

    # Ensure that the number of POS tags and labels match
    if len(clean_pos_tags) != len(labels):
        raise ValueError("Mismatch between POS tags length and label length.")
    
    return clean_pos_tags, labels


# Prepare training data using POS tags
def prepare_training_data(texts):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text in texts:
        # Tokenize and get binary labels
        clean_tokens, label_sequence = get_binary_pause_labels(text)
        
        # Tokenize with Camembert tokenizer using POS tags
        tokens = tokenizer(clean_tokens, is_split_into_words=True, padding="max_length", max_length=128, truncation=True)
        
        # Align the labels to tokens
        word_ids = tokens.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Special tokens
            elif word_id != previous_word_id:
                aligned_labels.append(label_sequence[word_id])  # Binary label (0 or 1)
            else:
                aligned_labels.append(label_sequence[word_id])  # Extend label to subwords
            previous_word_id = word_id
        
        # Pad the labels to match sequence length
        while len(aligned_labels) < 128:
            aligned_labels.append(-100)
        
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        labels.append(aligned_labels)

    return Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    })

# Function to get texts from a directory
def get_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, "r") as file:
            texts.append(file.read())
    return texts

# Path to your directory with text files
texts = get_texts("/Users/madalina/Documents/M1TAL/stage_GC/fichiersavecpausescat")

tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')
train_dataset = prepare_training_data(texts)

# Function to balance the dataset
def balance_dataset(train_dataset, oversample_factor=2, undersample_factor=2):
    pause_sequences = []
    no_pause_sequences = []
    
    for i in range(len(train_dataset)):
        if 1 in train_dataset[i]['labels']:  # Sequence contains at least one "pause" token
            pause_sequences.append(train_dataset[i])
        else:
            no_pause_sequences.append(train_dataset[i])

    # Oversample pause sequences
    oversampled_pause_sequences = pause_sequences * oversample_factor

    # Undersample no-pause sequences
    no_pause_sample_size = min(len(no_pause_sequences), len(pause_sequences) * undersample_factor)
    no_pause_sequences_balanced = random.sample(no_pause_sequences, k=no_pause_sample_size)

    # Combine into balanced dataset
    balanced_dataset = concatenate_datasets([
        Dataset.from_list(oversampled_pause_sequences + no_pause_sequences_balanced)
    ])

    return balanced_dataset

# Apply combined oversampling and undersampling
train_dataset = balance_dataset(train_dataset)

# Prepare dataset
train_size = int(0.8 * len(train_dataset))  # 80% for training
eval_size = len(train_dataset) - train_size  # 20% for evaluation

train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2).values()

# Load pre-trained Camembert for token classification with 2 labels (binary)
model = CamembertForTokenClassification.from_pretrained('camembert-base', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for checkpoints
    evaluation_strategy="steps",     # Evaluate every N steps
    save_steps=500,                  # Save checkpoint every N steps
    save_total_limit=2,              # Only keep last 2 checkpoints
    per_device_train_batch_size=8,   # Adjust batch size for your GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=20,              # Number of epochs
    learning_rate=2e-5,              # Learning rate
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Logging directory
    logging_steps=10,                # Log every 10 steps
    no_cuda=True                     # for MAC users
)

# Calculate class weights
pause_count = sum(label == 1 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
no_pause_count = sum(label == 0 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
total = pause_count + no_pause_count
class_weights = torch.tensor([total / no_pause_count, total / pause_count]).float()

# Custom Trainer with weighted loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Function to predict pauses based on POS tags
def predict_pause(text):
    pos_tags = get_pos_tags(text)
    tokens = tokenizer(pos_tags, is_split_into_words=True, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'].squeeze(), skip_special_tokens=True)
    prediction_labels = ["pause" if p == 1 else "no pause" for p in predictions]

    return list(zip(decoded_tokens, prediction_labels))

# Evaluation function
def evaluate_model(eval_dataset):
    metric = evaluate.load("accuracy")  # Load accuracy metric using evaluate
    model.eval()
    
    pause_correct, pause_total = 0, 0
    no_pause_correct, no_pause_total = 0, 0
    
    for batch in eval_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in batch.items() if k != 'labels'}
        labels = torch.tensor(batch['labels']).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1).view(-1)
        labels = labels.view(-1)

        mask = labels != -100
        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]
        
        for pred_label, true_label in zip(filtered_predictions.tolist(), filtered_labels.tolist()):
            if true_label == 1:
                pause_total += 1
                pause_correct += pred_label == 1
            else:
                no_pause_total += 1
                no_pause_correct += pred_label == 0
    
    pause_accuracy = (pause_correct / pause_total) if pause_total > 0 else 0
    no_pause_accuracy = (no_pause_correct / no_pause_total) if no_pause_total > 0 else 0

    print(f"Accuracy for 'pause': {pause_accuracy:.2f}")
    print(f"Accuracy for 'no pause': {no_pause_accuracy:.2f}")


accuracy = evaluate_model(eval_dataset)


def display_text_transformations(example_text, model, tokenizer):
    # 1. Print the original text
    print("Original Text:")
    print(example_text)
    
    # 2. Replace words with POS tags
    pos_tags = get_pos_tags(example_text)
    print("\nText with POS Tags:")
    print(" ".join(pos_tags))
    
    # 3. Get model predictions
    tokens = tokenizer(pos_tags, is_split_into_words=True, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'].squeeze(), skip_special_tokens=True)
    prediction_labels = ["pause" if p == 1 else "no pause" for p in predictions]

    print("\nPredictions:")
    for token, pos_tag, prediction in zip(decoded_tokens, pos_tags, prediction_labels):
        print(f"{token} ({pos_tag}): {prediction}")

sample_text = "Cependant, cat_4  pour les personnes ne voyant tout d'abord aucun avantage personnel cat_4 , ne trouve cat_1 nt aucune satisfaction  cat_5 dans celle-ci. Ainsi, cela peut entrainer un sentiment de rejet et d'incompréhension. cat_5 "
display_text_transformations(sample_text, model, tokenizer)

from sklearn.metrics import classification_report

# Function to compute evaluation metrics
def compute_metrics(eval_dataset, model, tokenizer):
    true_labels = []
    pred_labels = []

    model.eval()

    for batch in eval_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in batch.items() if k != 'labels'}
        labels = torch.tensor(batch['labels']).unsqueeze(0)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1).view(-1)
        labels = labels.view(-1)

        # Filter out padding labels (-100)
        mask = labels != -100
        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]

        pred_labels.extend(filtered_predictions.tolist())
        true_labels.extend(filtered_labels.tolist())

    # Print classification report
    report = classification_report(true_labels, pred_labels, target_names=['no pause', 'pause'], digits=4)
    print(report)
    return report

metrics_report = compute_metrics(eval_dataset, model, tokenizer)