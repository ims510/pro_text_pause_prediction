import re
import random
import torch
import pandas as pd
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, load, concatenate_datasets
import evaluate
import os
from torch.nn import CrossEntropyLoss

# Function to manually tokenize the text
def manual_tokenize(text):
    # Simple word-level tokenization,
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)  # Match words and punctuation
    return tokens

#Update the label extraction to binary classification (pause or no pause)
def get_binary_pause_labels(text):
    words = text.split()
    clean_tokens = []
    labels = []

    for word in words:
        # Add cleaned word to the tokens (handling categories separately)
        if "cat_" in word:
            # If there's a category marker, we mark the previous word with a pause
            if len(labels) > 0:  # Check that there is a previous word
                labels[-1] = 1  # Mark the previous word as followed by a pause
        else:
            clean_tokens.append(word)  # Add the actual word to clean tokens
            labels.append(0)  # No pause by default after this word
    
    # Ensure the label list and clean_tokens are aligned in length
    if len(clean_tokens) != len(labels):
        raise ValueError("Mismatch between token length and label length.")
    
    return clean_tokens, labels


def prepare_training_data(texts):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text in texts:
        # Tokenize and get binary labels
        clean_tokens, label_sequence = get_binary_pause_labels(text)
        
        # Tokenize with Camembert tokenizer
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

################# CHANGE THIS PATH TO YOUR DIRECTORY #################
# Texts with cat_1, cat_2, cat_3, cat_4, cat_5
texts = get_texts("/Users/madalina/Documents/M1TAL/stage_GC/fichiersavecpausescat")
print(texts)

tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')
train_dataset = prepare_training_data(texts)
print("Dataset:", train_dataset)

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
# Split the dataset into train and eval 
# Split the dataset into train (80%), validation (10%), and test (10%)
train_size = 0.8
val_test_size = 0.2  # Remaining 20% split into validation and test

# First, split the dataset into train and remaining (validation + test)
train_dataset, remaining_dataset = train_dataset.train_test_split(test_size=val_test_size).values()

# Now split the remaining dataset into validation and test (50% each from remaining 20%)
validation_dataset, test_dataset = remaining_dataset.train_test_split(test_size=0.5).values()

# Print out the sizes for confirmation
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


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
    load_best_model_at_end=True,    # Load the best model when finished
    no_cuda=True                    # for MAC users    
)

pause_count = sum(label == 1 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
no_pause_count = sum(label == 0 for label_seq in train_dataset['labels'] for label in label_seq if label != -100)
print(f"Pause count: {pause_count}, No pause count: {no_pause_count}")
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
# Train the model
trainer.train()

# Function to predict pauses in a given text
def predict_pause(text):
    # Tokenize the input text
    clean_tokens, _ = get_binary_pause_labels(text)
    tokens = tokenizer(clean_tokens, is_split_into_words=True, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)

    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # Convert tokens back to words and associate predictions
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'].squeeze(), skip_special_tokens=True)
    prediction_labels = ["pause" if p == 1 else "no pause" for p in predictions]

    return list(zip(decoded_tokens, prediction_labels))

def evaluate_model(eval_dataset):
    metric = evaluate.load("accuracy")  # Load accuracy metric using evaluate
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    pause_correct = 0
    pause_total = 0
    no_pause_correct = 0
    no_pause_total = 0
    
    for batch in eval_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in batch.items() if k != 'labels'}
        labels = torch.tensor(batch['labels']).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Flatten predictions and labels
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        
        # Filter out ⁠-100 ⁠ labels
        mask = labels != -100
        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]
        
        # Convert predictions and labels back to text
        input_ids = inputs['input_ids'].squeeze().tolist()
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
        
        # Ensure the length of decoded_tokens matches the mask
        mask_length = mask.sum().item()  # Count number of valid labels
        decoded_tokens = decoded_tokens[:mask_length]  # Truncate decoded tokens to mask length
        
        # Calculate correct predictions for each category
        for pred_label, true_label in zip(filtered_predictions.tolist(), filtered_labels.tolist()):
            if true_label == 1:  # If the true label is "pause"
                pause_total += 1
                if pred_label == 1:
                    pause_correct += 1
            elif true_label == 0:  # If the true label is "no pause"
                no_pause_total += 1
                if pred_label == 0:
                    no_pause_correct += 1
    
    # Calculate accuracy for each category
    pause_accuracy = (pause_correct / pause_total) if pause_total > 0 else 0
    no_pause_accuracy = (no_pause_correct / no_pause_total) if no_pause_total > 0 else 0
    
    print(f"Accuracy for 'pause': {pause_accuracy:.2f}")
    print(f"Accuracy for 'no pause': {no_pause_accuracy:.2f}")
    
    # Compute and return overall accuracy
    overall_accuracy = metric.compute(predictions=all_predictions, references=all_labels)
    return {
        "pause_accuracy": pause_accuracy,
        "no_pause_accuracy": no_pause_accuracy,
        "overall_accuracy": overall_accuracy
    }






# Evaluate the model and print detailed token predictions
print("\nDetailed Evaluation Predictions:")
accuracy = evaluate_model(test_dataset)
print("\nEvaluation Accuracy:", accuracy)

# import pickle
# import torch

# # Save the model
# torch.save(model.state_dict(), 'model.pth')

# # Save the tokenizer
# with open('tokenizer.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)

# # Save datasets
# with open('train_dataset.pkl', 'wb') as f:
#     pickle.dump(train_dataset, f)

# with open('eval_dataset.pkl', 'wb') as f:
#     pickle.dump(eval_dataset, f)

# # Save trainer logs for metrics
# with open('trainer_logs.pkl', 'wb') as f:
#     pickle.dump(trainer.state.log_history, f)

# # Example prediction
# example_text = "L'intention de l cat_5 'aéroport de biard cat_4 diminuer la poussé des gaz cat_1 sur le décollage de ses avions."
# predictions = predict_pause(example_text)
# print("Predictions for example text:")
# for token, label in predictions:
#     print(f"Token: {token}, Predicted: {label}")