import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import os
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

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

def prepare_t5_data(texts):
    inputs = []
    targets = []
    
    for text in texts:
        clean_tokens, label_sequence = get_binary_pause_labels(text)
        # Format input for T5
        inputs.append("tokenize: " + " ".join(clean_tokens))
        # Format output as space-separated binary labels
        targets.append(" ".join(map(str, label_sequence)))
    
    return Dataset.from_dict({'input_text': inputs, 'target_text': targets})

# Function to load texts from a directory
def get_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, "r") as file:
            texts.append(file.read())
    return texts

texts = get_texts("/Users/madalina/Documents/M1TAL/stage_GC/fichiersavecpausescat")
print(texts)

# Use T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


# Prepare dataset
dataset = prepare_t5_data(texts)

# Preprocess the data for T5
def preprocess_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function)

# Split dataset
train_size = 0.8
train_test_split = tokenized_dataset.train_test_split(test_size=1 - train_size)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,  # Load the best model
    no_cuda=True,                 # Use CPU for training if no GPU
)


# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

def evaluate_model(trainer, test_dataset):
    # Generate predictions
    raw_preds = trainer.predict(test_dataset)

    # Extract logits from the tuple
    predictions = raw_preds.predictions[0] if isinstance(raw_preds.predictions, tuple) else raw_preds.predictions

    # Take argmax over the vocab dimension (axis=-1)
    predictions = predictions.argmax(axis=-1)  # Shape: (batch_size, sequence_length)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(test_dataset["labels"], skip_special_tokens=True)

    # Print decoded predictions and labels for debugging
    print("Decoded Predictions:", decoded_preds)
    print("Decoded Labels:", decoded_labels)

    # Ensure predictions and labels are binary
    preds_flat = []
    labels_flat = []
    for preds in decoded_preds:
        for p in preds.split():
            if p.isdigit() and int(p) in [0, 1]:  # Ensure only binary values are included
                preds_flat.append(int(p))
    for labels in decoded_labels:
        for l in labels.split():
            if l.isdigit() and int(l) in [0, 1]:  # Ensure only binary values are included
                labels_flat.append(int(l))

    # Ensure the lengths of predictions and labels match
    min_length = min(len(preds_flat), len(labels_flat))
    preds_flat = preds_flat[:min_length]
    labels_flat = labels_flat[:min_length]

    # Calculate accuracy
    accuracy = accuracy_score(labels_flat, preds_flat)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate a classification report
    report = classification_report(labels_flat, preds_flat, target_names=["No Pause", "Pause"])
    print("\nClassification Report:\n", report)

    return accuracy, report


# Evaluate the model
accuracy, report = evaluate_model(trainer, test_dataset)

# Plot the classification report
def plot_classification_report(report):
    lines = report.split("\n")
    categories, precision, recall, f1_score = [], [], [], []

    for line in lines:
        tokens = line.split()
        if len(tokens) == 5 and tokens[0] not in ["accuracy", "macro", "weighted"]:
            categories.append(tokens[0])
            precision.append(float(tokens[1]))
            recall.append(float(tokens[2]))
            f1_score.append(float(tokens[3]))

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

plot_classification_report(report)
