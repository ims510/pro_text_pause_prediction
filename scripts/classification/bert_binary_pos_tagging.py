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
import spacy
from sklearn.preprocessing import OneHotEncoder

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

nlp = spacy.load("fr_core_news_sm")

# OneHotEncoder for POS vectorization
pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
encoder = OneHotEncoder(sparse_output=False)
encoder.fit([[pos] for pos in pos_list])

# Function to get POS tags and vectorize them
def get_pos_vectors(tokens):
    # Extract POS tags from tokens
    pos_tags = [[token.pos_] for token in tokens]  # Each POS tag needs to be a list
    # Use the encoder to transform POS tags into one-hot vectors
    pos_vectors = encoder.transform(pos_tags)  # Transform POS tags to one-hot encoded vectors
    return pos_vectors

# Prepare dataset for training with POS tagging and vectorization
def prepare_training_data(texts, tokenizer, max_length=128):
    input_ids, attention_masks, labels, pos_embeddings = [], [], [], []

    for idx, text in enumerate(texts):
        clean_tokens, label_sequence = get_binary_pause_labels(text)

        # Tokenize text using spaCy for POS tagging
        doc = nlp(" ".join(clean_tokens))

        # Tokenize with Camembert tokenizer
        tokens = tokenizer(clean_tokens, is_split_into_words=True, padding="max_length", max_length=max_length, truncation=True)
        word_ids = tokens.word_ids()

        # Get one-hot encoded POS vectors
        pos_vectors = get_pos_vectors(doc)

        aligned_labels, aligned_pos_vectors = [], []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Special tokens
                aligned_pos_vectors.append([0] * len(pos_list))  # Zero vector for special tokens
            elif word_id != previous_word_id:
                aligned_labels.append(label_sequence[word_id])
                aligned_pos_vectors.append(pos_vectors[word_id])  # POS vector for the word
            else:
                aligned_labels.append(label_sequence[word_id])
                aligned_pos_vectors.append(pos_vectors[word_id])  # Repeat POS vector for subwords
            previous_word_id = word_id

        # Padding labels and POS vectors to max_length
        aligned_labels += [-100] * (max_length - len(aligned_labels))
        while len(aligned_pos_vectors) < max_length:
            aligned_pos_vectors.append([0] * len(pos_list))  # Zero vector for padding

        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        labels.append(aligned_labels)
        pos_embeddings.append(aligned_pos_vectors)

        # Debug: Ensure pos_embeddings are added correctly for each sample
        print(f"Sample {idx}:")
        print(f"input_ids: {tokens['input_ids']}")
        print(f"labels: {aligned_labels}")
        print(f"pos_embeddings: {aligned_pos_vectors}")
        print('-' * 50)

        # Ensure pos_embeddings exists for all samples
        if not pos_embeddings[-1]:
            raise ValueError(f"Missing pos_embeddings in sample {idx}")

    dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'pos_embeddings': pos_embeddings  # Include POS embeddings in the dataset
    })
    
    # Debug: Print the features of the dataset
    print("Dataset features:", dataset.features)

    return dataset

def forward_pass_with_pos(model, inputs, pos_embeddings):
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state

    # Ensure pos_embeddings is of the same shape as token_embeddings
    pos_embeddings_tensor = torch.tensor(pos_embeddings).float().to(token_embeddings.device)

    # Ensure dimensions match: [batch_size, seq_length, pos_embedding_size]
    if pos_embeddings_tensor.shape[1] != token_embeddings.shape[1]:
        raise ValueError(f"Shape mismatch between token and pos embeddings: {token_embeddings.shape} vs {pos_embeddings_tensor.shape}")
    
    concatenated_embeddings = torch.cat((token_embeddings, pos_embeddings_tensor), dim=-1)
    return concatenated_embeddings

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

from transformers.modeling_outputs import TokenClassifierOutput

class CustomTrainerWithPOS(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print("Inputs keys:", inputs.keys())  # Check if 'pos_embeddings' is in inputs

        labels = inputs.pop("labels")
        pos_embeddings = inputs.pop("pos_embeddings")

        # Forward pass with concatenated POS embeddings
        concatenated_embeddings = forward_pass_with_pos(model, inputs, pos_embeddings)

        # Compute logits using the model's classifier on the concatenated embeddings
        logits = model.classifier(concatenated_embeddings)

        # Compute loss with CrossEntropy
        loss_fn = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))

        if return_outputs:
            return loss, TokenClassifierOutput(logits=logits)
        else:
            return loss

from transformers import DataCollatorForTokenClassification

class DataCollatorWithPosEmbeddings(DataCollatorForTokenClassification):
    def __call__(self, features):
        # Debug: Ensure pos_embeddings is present
        for idx, feature in enumerate(features):
            if 'pos_embeddings' not in feature:
                raise ValueError(f"Missing pos_embeddings in feature {idx}")

        # Extract pos_embeddings
        pos_embeddings = [feature.pop('pos_embeddings') for feature in features]
        
        # Convert pos_embeddings to a tensor
        pos_embeddings = torch.tensor(pos_embeddings)

        # Call the parent class's __call__ method to process the other elements
        batch = super().__call__(features)
        
        # Add pos_embeddings to the batch
        batch['pos_embeddings'] = pos_embeddings
        
        # Debugging output
        print("Batch created by DataCollator:", batch)
        
        return batch

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

# Verify that pos_embeddings are correctly added to the dataset
# for i, example in enumerate(train_dataset.select(range(5))):
#     print(f"Sample {i}:")
#     print("input_ids:", example['input_ids'])
#     print("labels:", example['labels'])
#     print("pos_embeddings:", example['pos_embeddings'])  # Check that this exists and looks correct

# Define class weights based on class imbalance
pause_count = sum(label == 1 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
no_pause_count = sum(label == 0 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
total = pause_count + no_pause_count
class_weights = torch.tensor([total / no_pause_count, total / pause_count]).float()

# Create a custom data collator with POS embeddings
data_collator = DataCollatorWithPosEmbeddings(tokenizer)

trainer = CustomTrainerWithPOS(
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
    data_collator=data_collator,  # Custom data collator with POS embeddings
)

# Ensure pos_embeddings are in each batch in the Dataset
print(train_dataset.features)

# Train the model
trainer.train()

# Evaluate the model
evaluate_model(eval_dataset, model, tokenizer)
# Evaluate the model
evaluate_model(eval_dataset, model, tokenizer)
