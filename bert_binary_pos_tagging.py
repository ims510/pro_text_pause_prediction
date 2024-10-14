import os
import re
import random
import torch
import pandas as pd
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, concatenate_datasets
import evaluate
from torch.nn import CrossEntropyLoss

# Manual tokenization function (using regex to match words and punctuation)
def manual_tokenize(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

# Extract binary pause labels from text
def get_binary_pause_labels(text):
    words = text.split()
    clean_tokens, labels = [], []

    for word in words:
        if "cat_" in word:
            if labels:
                labels[-1] = 1  # Mark previous word with a pause
        else:
            clean_tokens.append(word)
            labels.append(0)  # Default to no pause after word

    if len(clean_tokens) != len(labels):
        raise ValueError("Mismatch between tokens and labels.")
    
    return clean_tokens, labels

# Prepare dataset for training
def prepare_training_data(texts, tokenizer, max_length=128):
    input_ids, attention_masks, labels = [], [], []
    
    for text in texts:
        clean_tokens, label_sequence = get_binary_pause_labels(text)
        tokens = tokenizer(clean_tokens, is_split_into_words=True, padding="max_length", max_length=max_length, truncation=True)
        
        word_ids = tokens.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Ignore special tokens
            elif word_id != previous_word_id:
                aligned_labels.append(label_sequence[word_id])
            else:
                aligned_labels.append(label_sequence[word_id])
            previous_word_id = word_id

        # Padding labels
        aligned_labels += [-100] * (max_length - len(aligned_labels))
        
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        labels.append(aligned_labels)

    return Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    })

# Get texts from directory
def get_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            texts.append(file.read())
    return texts

# Function to balance dataset with oversampling/undersampling
def balance_dataset(train_dataset, oversample_factor=2, undersample_factor=2):
    pause_sequences, no_pause_sequences = [], []

    for i in range(len(train_dataset)):
        if 1 in train_dataset[i]['labels']:
            pause_sequences.append(train_dataset[i])
        else:
            no_pause_sequences.append(train_dataset[i])

    oversampled_pause_sequences = pause_sequences * oversample_factor
    no_pause_sample_size = min(len(no_pause_sequences), len(pause_sequences) * undersample_factor)
    no_pause_sequences_balanced = random.sample(no_pause_sequences, k=no_pause_sample_size)

    balanced_dataset = concatenate_datasets([Dataset.from_list(oversampled_pause_sequences + no_pause_sequences_balanced)])
    return balanced_dataset

# Setup for training and evaluation
def setup_training(tokenizer, model, texts, training_args):
    train_dataset = prepare_training_data(texts, tokenizer)
    train_dataset = balance_dataset(train_dataset)

    train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2).values()

    return train_dataset, eval_dataset

# Define CustomTrainer class to handle weighted loss for imbalanced classes
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Evaluation function for pause/no pause accuracy
def evaluate_model(eval_dataset, model, tokenizer):
    metric = evaluate.load("accuracy")
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

    return {
        "pause_accuracy": pause_accuracy,
        "no_pause_accuracy": no_pause_accuracy
    }

# Main script to run training and evaluation
texts = get_texts("/Users/madalina/Documents/M1TAL/stage_GC/fichiersavecpausescat")
tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')
model = CamembertForTokenClassification.from_pretrained('camembert-base', num_labels=2)

train_dataset, eval_dataset = setup_training(tokenizer, model, texts, TrainingArguments(
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
))

# Define class weights based on class imbalance
pause_count = sum(label == 1 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
no_pause_count = sum(label == 0 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
total = pause_count + no_pause_count
class_weights = torch.tensor([total / no_pause_count, total / pause_count]).float()

trainer = CustomTrainer(
    model=model,
    args=TrainingArguments(
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
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    class_weights=class_weights
)

# Train the model
trainer.train()

# Evaluate the model
evaluate_model(eval_dataset, model, tokenizer)
