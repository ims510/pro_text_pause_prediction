import re
import random
import torch
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, load, concatenate_datasets
import evaluate
import os
from torch.nn import CrossEntropyLoss


import keras
import tensorflow as tf
from transformers import __version__

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
print("Transformers version:", __version__)

# Function to manually tokenize the text
def manual_tokenize(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

# Update the label extraction to binary classification (pause or no pause)
def get_binary_pause_labels(text):
    words = text.split()
    clean_tokens = []
    labels = []

    for word in words:
        if "cat_" in word:
            if len(labels) > 0:
                labels[-1] = 1
        else:
            clean_tokens.append(word)
            labels.append(0)
    
    if len(clean_tokens) != len(labels):
        raise ValueError("Mismatch between token length and label length.")
    
    return clean_tokens, labels

def prepare_training_data(texts):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text in texts:
        clean_tokens, label_sequence = get_binary_pause_labels(text)
        tokens = tokenizer(clean_tokens, is_split_into_words=True, padding="max_length", max_length=128, truncation=True)
        
        word_ids = tokens.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(label_sequence[word_id])
            else:
                aligned_labels.append(label_sequence[word_id])
            previous_word_id = word_id
        
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

def get_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, "r") as file:
            texts.append(file.read())
    return texts

texts = get_texts("/Users/madalina/Documents/M1TAL/stage_GC/fichiersavecpausescat")
print(texts)

# Use Roberta tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
train_dataset = prepare_training_data(texts)
print("Dataset:", train_dataset)

def balance_dataset(train_dataset, oversample_factor=2, undersample_factor=2):
    pause_sequences = []
    no_pause_sequences = []
    
    for i in range(len(train_dataset)):
        if 1 in train_dataset[i]['labels']:
            pause_sequences.append(train_dataset[i])
        else:
            no_pause_sequences.append(train_dataset[i])

    oversampled_pause_sequences = pause_sequences * oversample_factor
    no_pause_sample_size = min(len(no_pause_sequences), len(pause_sequences) * undersample_factor)
    no_pause_sequences_balanced = random.sample(no_pause_sequences, k=no_pause_sample_size)

    balanced_dataset = concatenate_datasets([
        Dataset.from_list(oversampled_pause_sequences + no_pause_sequences_balanced)
    ])

    return balanced_dataset

train_dataset = balance_dataset(train_dataset)

train_size = 0.8
val_test_size = 0.2

train_dataset, remaining_dataset = train_dataset.train_test_split(test_size=val_test_size).values()
validation_dataset, test_dataset = remaining_dataset.train_test_split(test_size=0.5).values()

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Load pre-trained RoBERTa for token classification
model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    no_cuda=True
)

pause_count = sum(label == 1 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
no_pause_count = sum(label == 0 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
total = pause_count + no_pause_count
class_weights = torch.tensor([total / no_pause_count, total / pause_count]).float()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)

trainer.train()

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(trainer, test_dataset, tokenizer):
    # Evaluate the model
    raw_preds = trainer.predict(test_dataset)
    predictions = torch.argmax(torch.tensor(raw_preds.predictions), axis=-1)
    
    # Flatten predictions and labels, removing special tokens (-100)
    preds_flat = []
    labels_flat = []
    for pred, label in zip(predictions, test_dataset['labels']):
        for p, l in zip(pred, label):
            if l != -100:  # Exclude special tokens
                preds_flat.append(p.item())
                labels_flat.append(l)

    # Calculate accuracy
    accuracy = accuracy_score(labels_flat, preds_flat)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate a classification report
    report = classification_report(labels_flat, preds_flat, target_names=["No Pause", "Pause"])
    print("\nClassification Report:\n", report)
    
    return accuracy, report

def plot_classification_report(report):
    # Parse the classification report into lines
    lines = report.split("\n")
    
    # Initialize lists for categories and metrics
    categories = []
    precision = []
    recall = []
    f1_score = []

    # Process each line
    for line in lines:
        tokens = line.split()
        # Check if this line contains valid metrics (expect at least 4 tokens: category + metrics)
        if len(tokens) == 5 and tokens[0] not in ["accuracy", "macro", "weighted"]:  # Exclude summary lines
            try:
                categories.append(tokens[0])            # First token is the category
                precision.append(float(tokens[1]))      # Precision
                recall.append(float(tokens[2]))         # Recall
                f1_score.append(float(tokens[3]))       # F1-Score
            except ValueError:
                # Skip lines with invalid data
                continue

    # Create the bar plot
    x = range(len(categories))
    plt.figure(figsize=(8, 6))
    plt.bar(x, precision, width=0.2, label="Precision", align="center")
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label="Recall", align="center")
    plt.bar([p + 0.4 for p in x], f1_score, width=0.2, label="F1-Score", align="center")
    plt.xticks([p + 0.2 for p in x], categories)
    plt.xlabel("Categories")
    plt.ylabel("Scores")
    plt.title("Classification Report Metrics")
    plt.legend()
    plt.show()


# Evaluate the model
accuracy, report = evaluate_model(trainer, test_dataset, tokenizer)

# Plot the classification report
plot_classification_report(report)
